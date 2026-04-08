# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from copy import deepcopy
import logging
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.augmentations import apply_transform
from utils.conf import create_seeded_dataloader, get_device

if TYPE_CHECKING:
    from models.utils.continual_model import ContinualModel
    from datasets.utils.continual_dataset import ContinualDataset
    from backbone import MammothBackbone


def icarl_replay(self: 'ContinualModel', dataset: 'ContinualDataset', val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.

    Args:
        self: the model instance
        dataset: the dataset
        val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.current_task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_dataset = deepcopy(dataset.train_loader.dataset)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_dataset.targets = np.concatenate([
                self.val_dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_dataset.data = data_concatenate([
                self.val_dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])

            self.val_loader = create_seeded_dataloader(self.args, self.val_dataset, batch_size=self.args.batch_size, shuffle=True)


class BaseSampleSelection:
    """
    Base class for sample selection strategies.
    """

    def __init__(self, buffer_size: int, device):
        """
        Initialize the sample selection strategy.

        Args:
            buffer_size: the maximum buffer size
            device: the device to store the buffer on
        """
        self.buffer_size = buffer_size
        self.device = device

    def __call__(self, num_seen_examples: int) -> int:
        """
        Selects the index of the sample to replace.

        Args:
            num_seen_examples: the number of seen examples

        Returns:
            the index of the sample to replace
        """

        raise NotImplementedError

    def update(self, *args, **kwargs):
        """
        (optional) Update the state of the sample selection strategy.
        """
        pass


class ReservoirSampling(BaseSampleSelection):
    def __call__(self, num_seen_examples: int) -> int:
        """
        Reservoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < self.buffer_size:
            return rand
        else:
            return -1


class BalancoirSampling(BaseSampleSelection):
    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        self.unique_map = np.empty((0,), dtype=np.int32)

    def update_unique_map(self, label_in, label_out=None):
        while len(self.unique_map) <= label_in:
            self.unique_map = np.concatenate((self.unique_map, np.zeros((len(self.unique_map) * 2 + 1), dtype=np.int32)), axis=0)
        self.unique_map[label_in] += 1
        if label_out is not None:
            self.unique_map[label_out] -= 1

    def __call__(self, num_seen_examples: int, labels: torch.Tensor, proposed_class: int) -> int:
        """
        Balancoir sampling algorithm.

        Args:
            num_seen_examples: the number of seen examples
            buffer_size: the maximum buffer size
            labels: the set of buffer labels
            proposed_class: the class of the current example

        Returns:
            the target index if the current image is sampled, else -1
        """
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rand = np.random.randint(0, num_seen_examples + 1)
        if rand < self.buffer_size or len(self.unique_map) <= proposed_class or self.unique_map[proposed_class] < np.median(
                self.unique_map[self.unique_map > 0]):
            target_class = np.argmax(self.unique_map)
            # e = rand % self.unique_map.max()
            idx = np.arange(self.buffer_size)[labels.cpu() == target_class][rand % self.unique_map.max()]
            return idx
        else:
            return -1


class LARSSampling(BaseSampleSelection):
    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        # lossoir scores
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        self.importance_scores[indexes] = values

    def normalize_scores(self, values: torch.Tensor):
        if values.shape[0] > 0:
            if values.max() - values.min() != 0:
                values = (values - values.min()) / ((values.max() - values.min()) + 1e-9)
            return values
        else:
            return None

    def __call__(self, num_seen_examples: int) -> int:
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)
        if rn < self.buffer_size:
            norm_importance = self.normalize_scores(self.importance_scores)
            norm_importance = norm_importance / (norm_importance.sum() + 1e-9)
            index = np.random.choice(range(self.buffer_size), p=norm_importance.cpu().numpy(), size=1)
            return index
        else:
            return -1

class MaxEntSampling(BaseSampleSelection):
    """
    Maximum-Entropy weighted reservoir sampling.
    分数定义：预测分布的熵 H(p) = -Σ p_i log p_i （越大越不确定）。
    逻辑：当需要替换时，在 buffer 内按归一化的“熵分数”分布抽一个位置作为替换目标。
    """

    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        # 写入/更新被选中 index 的分数（由 Buffer.add_data 传入）
        self.importance_scores[indexes] = values

    def _normalize(self, values: torch.Tensor):
        if values.numel() == 0:
            return None
        vmin, vmax = values.min(), values.max()
        if torch.isinf(vmin) and torch.isinf(vmax):
            # 全是 -inf（初始化阶段），退化为均匀
            return torch.full_like(values, 1.0 / len(values))
        rng = (vmax - vmin).clamp_min(1e-9)
        norm = (values - vmin) / rng
        s = norm.sum().clamp_min(1e-9)
        return norm / s

    def __call__(self, num_seen_examples: int) -> int:
        # 先按 reservoir 门控，避免过度替换早期样本（与 LARS/LABRS 的写法一致）
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)  # [0, num_seen_examples-1]
        if rn < self.buffer_size:
            probs = self._normalize(self.importance_scores)
            inv = (1.0 - probs).clamp_min(1e-9)
            probs = inv / inv.sum()
            index = int(np.random.choice(range(self.buffer_size),
                                         p=probs.detach().cpu().numpy(),
                                         size=1)[0])
            return index
        else:
            return -1


class LossAwareBalancedSampling(BaseSampleSelection):
    """
    Combination of Loss-Aware Sampling (LARS) and Balanced Reservoir Sampling (BRS) from `Rethinking Experience Replay: a Bag of Tricks for Continual Learning`.
    """

    def __init__(self, buffer_size: int, device):
        super().__init__(buffer_size, device)
        # lossoir scores
        self.importance_scores = torch.ones(buffer_size, device=device) * -float('inf')
        # balancoir scores
        self.balance_scores = torch.ones(self.buffer_size, dtype=torch.float).to(self.device) * -float('inf')
        # merged scores
        self.scores = torch.ones(self.buffer_size).to(self.device) * -float('inf')

    def update(self, indexes: torch.Tensor, values: torch.Tensor):
        self.importance_scores[indexes] = values

    def merge_scores(self):
        scaling_factor = self.importance_scores.abs().mean() * self.balance_scores.abs().mean()
        norm_importance = self.importance_scores / scaling_factor
        presoftscores = 0.5 * norm_importance + 0.5 * self.balance_scores

        if presoftscores.max() - presoftscores.min() != 0:
            presoftscores = (presoftscores - presoftscores.min()) / (presoftscores.max() - presoftscores.min() + 1e-9)
        self.scores = presoftscores / presoftscores.sum()

    def update_balancoir_scores(self, labels: torch.Tensor):
        unique_labels, orig_inputs_idxs, counts = labels.unique(return_counts=True, return_inverse=True)
        # assert len(counts) > unique_labels.max(), "Some classes are missing from the buffer"
        self.balance_scores = torch.gather(counts, 0, orig_inputs_idxs).float()

    def __call__(self, num_seen_examples: int, labels: torch.Tensor) -> int:
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)
        if rn < self.buffer_size:
            self.update_balancoir_scores(labels)
            self.merge_scores()
            index = np.random.choice(range(self.buffer_size), p=self.scores.cpu().numpy(), size=1)
            return index
        else:
            return -1


class ABSSampling(LARSSampling):
    def __init__(self, buffer_size: int, device: str, dataset: 'ContinualDataset'):
        super().__init__(buffer_size, device)
        self.dataset = dataset

    def scale_scores(self, past_indexes: torch.Tensor):
        # due normalizzazioni divere per i due gruppi
        past_importance = self.normalize_scores(self.importance_scores[past_indexes])
        current_importance = self.normalize_scores(self.importance_scores[~past_indexes])
        current_scores, past_scores = None, None
        if past_importance is not None:
            past_importance = 1 - past_importance
            past_scores = past_importance / past_importance.sum()
        if current_importance is not None:
            if current_importance.sum() == 0:
                current_importance += 1e-9
            current_scores = current_importance / current_importance.sum()

        return past_scores, current_scores

    def __call__(self, num_seen_examples: int, labels: torch.Tensor) -> int:
        n_seen_classes, _ = self.dataset.get_offsets()

        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        rn = np.random.randint(0, num_seen_examples)
        if rn < self.buffer_size:
            past_indexes = labels < n_seen_classes

            past_scores, current_scores = self.scale_scores(past_indexes)
            past_percentage = np.float64(past_indexes.sum().cpu() / self.buffer_size)  # avoid numerical issues
            pres_percetage = 1 - past_percentage
            assert past_percentage + pres_percetage == 1, f"The sum of the percentages must be 1 but found {past_percentage+pres_percetage}: {past_percentage} + {pres_percetage}"
            rp = np.random.choice((0, 1), p=[past_percentage, pres_percetage])

            if not rp:
                index = np.random.choice(np.arange(self.buffer_size)[past_indexes.cpu().numpy()], p=past_scores.cpu().numpy(), size=1)
            else:
                index = np.random.choice(np.arange(self.buffer_size)[~past_indexes.cpu().numpy()], p=current_scores.cpu().numpy(), size=1)
            return index
        else:
            return -1

class AngleMidSampling(BaseSampleSelection):
    """
    Angle-driven replacement with per-class quota (方案 A):
    - 目标：全局保证“每个已见类至少达到配额 quota = buffer_size // n_seen_classes”，
      同时在替换时优先“保留更接近类内中位角（mid-angle）的样本”。

    规则（buffer 已满时）：
    1) 若新样本所属类 c 的计数 cnt(c) < quota：
       - 允许“跨类替换”：在超额类（cnt > quota）中，找到一个“最不 mid”的样本踢掉；
       - 若不存在超额类，则退回 reservoir 风格兜底。
    2) 否则（cnt(c) >= quota）：
       - 在“类 c 内部”比较 mid-angle：若新样本更 mid，则替换掉该类中“最不 mid”的样本；
         否则丢弃。

    依赖：
    - 需要 net（backbone）与归一化变换 norm_transform（或恒等）。
    - 需要 labels（用于定位类与统计类计数）。
    - 需要访问 buffer（读取 buffer.examples / buffer.labels）。
    """

    def __init__(self, buffer_size: int, device, net: 'MammothBackbone', norm_transform=None, min_quota: int = 1):
        super().__init__(buffer_size, device)
        self.net = net
        self.norm_transform = norm_transform
        self.min_quota = max(1, int(min_quota))

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W)
        先做归一化变换，再过 backbone 抽 features（不切 classifier），最后做 L2 归一化。
        """
        if self.norm_transform is None:
            def _norm(z): return z
        else:
            def _norm(z): return self.norm_transform(z)

        was_training = self.net.training
        self.net.eval()
        try:
            x_dev = x.to(self.device)
            x_dev = _norm(x_dev)
            feats = self.net(x_dev, returnt='features')
        finally:
            self.net.train(was_training)

        # feats = F.normalize(feats, dim=1)
        return feats

    @torch.no_grad()
    def _worst_mid_index_in_indexset(self, buffer: 'Buffer', index_set: torch.Tensor) -> int:
        """
        在给定的 buffer 索引集合中，找出“距离类内中位角最远”（最不 mid）的样本的全局索引。
        """
        if index_set.numel() == 1:
            return index_set.item()

        ex = buffer.examples[index_set]
        feats = self._extract_features(ex)
        mean_feat = F.normalize(feats.mean(0, keepdim=True), dim=1)
        feats = F.normalize(feats, dim=1)
        cos_sim = F.cosine_similarity(feats, mean_feat, dim=1)
        med = cos_sim.median()
        dist = (cos_sim - med).abs()
        worst_local = dist.argmax().item()
        return index_set[worst_local].item()

    @torch.no_grad()
    def __call__(self, num_seen_examples: int, *,
                 buffer: 'Buffer',
                 example: torch.Tensor,
                 example_label: torch.Tensor) -> int:
        """
        返回：要写入的位置；-1 表示丢弃新样本。
        以关键字参数形式接收上下文，Buffer.add_data 会传入。
        """
        # 1) 未满：顺序填充
        if num_seen_examples < self.buffer_size:
            return num_seen_examples

        # 2) 无标签或 buffer 无 labels：退回 reservoir 兜底
        if example_label is None or not hasattr(buffer, 'labels'):
            rn = np.random.randint(0, num_seen_examples + 1)
            return rn if rn < self.buffer_size else -1

        # 当前有效长度与标签
        cur_len = min(buffer.num_seen_examples, buffer.examples.shape[0])
        buf_labels = buffer.labels[:cur_len]

        # 已见类集合与计数
        unique, counts = buf_labels.unique(return_counts=True)

        # 已见类数（若新类尚未在 buffer 出现，则 +1 计入）
        y_val = int(example_label.item())
        seen_set = set(unique.tolist())
        n_seen = len(seen_set) + (0 if y_val in seen_set else 1)
        quota = max(self.min_quota, self.buffer_size // max(1, n_seen))

        # 当前类计数
        cls_cnt = int((buf_labels == y_val).sum().item())

        # --------- 情况 A：当前类未达配额 → 跨类替换 ---------
        if cls_cnt < quota:
            # 找超额类（cnt > quota）
            over_mask = counts > quota
            if over_mask.any():
                # 选择“超额最多”的类作为 victim（也可随机超额类）
                over_classes = unique[over_mask]
                over_counts = counts[over_mask]
                victim_class = int(over_classes[(over_counts - quota).argmax()].item())

                # victim 类内找“最不 mid”的一个样本并替掉
                victim_idx = (buf_labels == victim_class).nonzero(as_tuple=False).squeeze(1)
                return self._worst_mid_index_in_indexset(buffer, victim_idx)

            # 没有超额类时，退回 reservoir 兜底（避免饿死）
            rn = np.random.randint(0, num_seen_examples + 1)
            return rn if rn < self.buffer_size else -1

        # --------- 情况 B：当前类已达配额 → 同类内 mid-angle 替换 ---------
        # 找同类索引
        same_cls_idx = (buf_labels == y_val).nonzero(as_tuple=False).squeeze(1)
        if same_cls_idx.numel() == 0:
            # 理论上不会发生（cls_cnt >= quota），但保底处理
            rn = np.random.randint(0, num_seen_examples + 1)
            return rn if rn < self.buffer_size else -1

        # 现有同类特征
        buf_examples_same = buffer.examples[same_cls_idx]
        feats_old = self._extract_features(buf_examples_same)   # (K, D)

        # 新样本特征
        feat_new = self._extract_features(example.unsqueeze(0)) # (1, D)

        # 与类内均值的中位角距离比较
        feats_all = torch.cat([feats_old, feat_new], dim=0)     # (K+1, D)
        mean_feat = F.normalize(feats_all.mean(0, keepdim=True), dim=1)
        feats_all = F.normalize(feats_all, dim=1)
        cos_sim_all = F.cosine_similarity(feats_all, mean_feat, dim=1)
        cos_med = cos_sim_all.median()
        dist_all = (cos_sim_all - cos_med).abs()
        dist_old = dist_all[:-1]
        dist_new = float(dist_all[-1].item())

        worst_pos_local = int(dist_old.argmax().item())
        worst_dist = float(dist_old[worst_pos_local].item())

        if dist_new < worst_dist:
            # 替掉本类中“最不 mid”的那个
            return int(same_cls_idx[worst_pos_local].item())
        else:
            return -1



class Buffer:
    """
    The memory buffer of rehearsal method.
    """

    buffer_size: int  # the maximum size of the buffer
    device: str  # the device to store the buffer on
    num_seen_examples: int  # the total number of examples seen, used for reservoir
    attributes: List[str]  # the attributes stored in the buffer
    attention_maps: List[torch.Tensor]  # (optional) attention maps used by TwF
    sample_selection_strategy: str  # the sample selection strategy used to select samples to replace. By default, 'reservoir'

    examples: torch.Tensor  # (mandatory) buffer attribute: the tensor of images
    labels: torch.Tensor  # (optional) buffer attribute: the tensor of labels
    logits: torch.Tensor  # (optional) buffer attribute: the tensor of logits
    task_labels: torch.Tensor  # (optional) buffer attribute: the tensor of task labels
    true_labels: torch.Tensor  # (optional) buffer attribute: the tensor of true labels

    def __init__(self, buffer_size: int, device="cpu", sample_selection_strategy='reservoir', **kwargs):
        """
        Initialize a reservoir-based Buffer object.

        Supports storing images, labels, logits, task_labels, and attention maps. This can be extended by adding more attributes to the `attributes` list and updating the `init_tensors` method accordingly.

        To select samples to replace, the buffer supports:
        - `reservoir` sampling: randomly selects samples to replace (default). Ref: "Jeffrey S Vitter. Random sampling with a reservoir."
        - `lars`: prioritizes retaining samples with the *higher* loss. Ref: "Pietro Buzzega et al. Rethinking Experience Replay: a Bag of Tricks for Continual Learning."
        - `labrs` (Loss-Aware Balanced Reservoir Sampling): combination of LARS and BRS. Ref: "Pietro Buzzega et al. Rethinking Experience Replay: a Bag of Tricks for Continual Learning."
        - `abs` (Asymmetric Balanced Sampling): for samples from the current task, prioritizes retaining samples with the *lower* loss (i.e., inverse `lossoir`); for samples from previous tasks, prioritizes retaining samples with the *higher* loss (i.e., `lossoir`). Useful for settings with noisy labels. Ref: "Monica Millunzi et al. May the Forgetting Be with You: Alternate Replay for Learning with Noisy Labels".

        Args:
            buffer_size (int): The maximum size of the buffer.
            device (str, optional): The device to store the buffer on. Defaults to "cpu".
            sample_selection_strategy: The sample selection strategy. Defaults to 'reservoir'. Options: 'reservoir', 'lars', 'labrs', 'abs', 'balancoir'.

        Note:
            If during the `get_data` the transform is PIL, data will be moved to cpu and then back to the device. This is why the device is set to cpu by default.
        """
        self._dl_transform = None
        self._it_index = 0
        self._buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'true_labels']
        self.attention_maps = [None] * buffer_size
        self.sample_selection_strategy = sample_selection_strategy

        assert sample_selection_strategy.lower() in ['reservoir', 'lars', 'labrs', 'abs', 'balancoir', 'unlimited', 'angle_mid', 'maxent'], f"Invalid sample selection strategy: {sample_selection_strategy}"

        if sample_selection_strategy.lower() == 'abs':
            assert 'dataset' in kwargs, "The dataset is required for ABS sample selection"
            self.sample_selection_fn = ABSSampling(buffer_size, device, kwargs['dataset'])
        elif sample_selection_strategy.lower() == 'lars':
            self.sample_selection_fn = LARSSampling(buffer_size, device)
        # ...
        elif sample_selection_strategy.lower() == 'maxent':
            self.sample_selection_fn = MaxEntSampling(buffer_size, device)
        elif sample_selection_strategy.lower() == 'labrs':
            self.sample_selection_fn = LossAwareBalancedSampling(buffer_size, device)
        elif sample_selection_strategy.lower() == 'unlimited':
            self.sample_selection_fn = lambda x: x
            self._buffer_size = 10  # initial buffer size, will be expanded if needed
        elif sample_selection_strategy.lower() == 'balancoir':
            self.sample_selection_fn = BalancoirSampling(buffer_size, device)
        elif sample_selection_strategy.lower() == 'angle_mid':
            # 需要传入 net 与 norm_transform。若未给，降级为 reservoir（稳妥）
            net = kwargs.get('net', None)
            norm_transform = kwargs.get('norm_transform', None)
            if net is not None:
                self.sample_selection_fn = AngleMidSampling(buffer_size, device, net=net, norm_transform=norm_transform)
            else:
                logging.warning("[Buffer] 'angle_mid' requires 'net'; falling back to reservoir.")
                self.sample_selection_fn = ReservoirSampling(buffer_size, device)
        else:
            self.sample_selection_fn = ReservoirSampling(buffer_size, device)

    def serialize(self, out_device='cpu'):
        """
        Serialize the buffer.

        Returns:
            A dictionary containing the buffer attributes.
        """
        return {attr_str: getattr(self, attr_str).to(out_device) for attr_str in self.attributes if hasattr(self, attr_str)}

    def to(self, device):
        """
        Move the buffer and its attributes to the specified device.

        Args:
            device: The device to move the buffer and its attributes to.

        Returns:
            The buffer instance with the updated device and attributes.
        """
        self.device = device
        self.sample_selection_fn.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        """
        Returns the number items in the buffer.
        """
        if self.sample_selection_strategy == 'unlimited':
            return self.num_seen_examples
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor,
                     true_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            true_labels: tensor containing the true labels (used only for logging)
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):  # create tensor if not already present
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self._buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))
            elif hasattr(self, attr_str):  # if tensor already exists, update it and possibly resize it according to the buffer_size
                if self.num_seen_examples < self._buffer_size:  # if the buffer is full, extend the tensor
                    old_tensor = getattr(self, attr_str)
                    pad = torch.zeros((self._buffer_size - old_tensor.shape[0], *attr.shape[1:]), dtype=old_tensor.dtype, device=self.device)
                    setattr(self, attr_str, torch.cat([old_tensor, pad], dim=0))

    @property
    def buffer_size(self):
        """
        Returns the buffer size.
        """
        if self.sample_selection_strategy == 'unlimited':
            # return max int if unlimited
            return int(1e9)
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value):
        """
        Sets the buffer size.
        """
        if self.sample_selection_strategy != 'unlimited':
            self._buffer_size = value

    @property
    def used_attributes(self):
        """
        Returns a list of attributes that are currently being used by the object.
        """
        return [attr_str for attr_str in self.attributes if hasattr(self, attr_str)]

    def is_full(self):
        return self.num_seen_examples >= self.buffer_size

    def add_data(self, examples, labels=None, logits=None, task_labels=None, attention_maps=None, true_labels=None, sample_selection_scores=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.

        Args:
            examples: tensor containing the images
            labels: tensor containing the labels
            logits: tensor containing the outputs of the network
            task_labels: tensor containing the task labels
            attention_maps: list of tensors containing the attention maps
            true_labels: if setting is noisy, the true labels associated with the examples. **Used only for logging.**
            sample_selection_scores: tensor containing the scores used for the sample selection strategy. NOTE: this is only used if the sample selection strategy defines the `update` method.

        Note:
            Only the examples are required. The other tensors are initialized only if they are provided.
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, true_labels)

        for i in range(examples.shape[0]):
            if self.sample_selection_strategy == 'abs' or self.sample_selection_strategy == 'labrs':
                index = self.sample_selection_fn(self.num_seen_examples, labels=self.labels)
            elif self.sample_selection_strategy == 'balancoir':
                index = self.sample_selection_fn(self.num_seen_examples, labels=self.labels, proposed_class=labels[i])
            elif self.sample_selection_strategy == 'angle_mid':
                # 关键：把 buffer / 当前样本 / 当前样本标签 传给策略
                cur_label = labels[i] if labels is not None else None
                index = self.sample_selection_fn(self.num_seen_examples,
                                                 buffer=self,
                                                 example=examples[i],
                                                 example_label=cur_label)
            else:
                index = self.sample_selection_fn(self.num_seen_examples)
            self.num_seen_examples += 1
            if index >= 0:
                if self.sample_selection_strategy == 'unlimited' and self.num_seen_examples > self._buffer_size:
                    self._buffer_size *= 2
                    self.init_tensors(examples, labels, logits, task_labels, true_labels)
                if self.sample_selection_strategy == 'balancoir':
                    self.sample_selection_fn.update_unique_map(labels[i], self.labels[index] if index < self.num_seen_examples else None)

                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if attention_maps is not None:
                    self.attention_maps[index] = [at[i].byte().to(self.device) for at in attention_maps]
                if sample_selection_scores is not None:
                    self.sample_selection_fn.update(index, sample_selection_scores[i])
                if true_labels is not None:
                    self.true_labels[index] = true_labels[i].to(self.device)

    def get_data(self, size: int, transform: nn.Module = None, return_index=False, device=None,
                 mask_task_out=None, cpt=None, return_not_aug=False, not_aug_transform=None, force_indexes=None) -> Tuple:
        """
        Random samples a batch of size items.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            return_index: if True, returns the indexes of the sampled items
            mask_task: if not None, masks OUT the examples from the given task
            cpt: the number of classes per task (required if mask_task is not None and task_labels are not present)
            return_not_aug: if True, also returns the not augmented items
            not_aug_transform: the transformation to be applied to the not augmented items (if `return_not_aug` is True)
            forced_indexes: if not None, forces the selection of the samples with the given indexes

        Returns:
            a tuple containing the requested items. If return_index is True, the tuple contains the indexes as first element.
        """
        target_device = self.device if device is None else device

        if mask_task_out is not None:
            assert hasattr(self, 'task_labels') or cpt is not None
            assert hasattr(self, 'task_labels') or hasattr(self, 'labels')
            samples_mask = (self.task_labels != mask_task_out) if hasattr(self, 'task_labels') else self.labels // cpt != mask_task_out

        num_avail_samples = self.examples.shape[0] if mask_task_out is None else samples_mask.sum().item()
        num_avail_samples = min(self.num_seen_examples, num_avail_samples)

        if size > min(num_avail_samples, self.examples.shape[0]):
            size = min(num_avail_samples, self.examples.shape[0])

        if force_indexes is not None:
            choice = force_indexes if isinstance(force_indexes, np.ndarray) else np.array(force_indexes)
        else:
            choice = np.random.choice(num_avail_samples, size=size, replace=False)
        if transform is None:
            def transform(x): return x

        selected_samples = self.examples[choice] if mask_task_out is None else self.examples[samples_mask][choice]

        if return_not_aug:
            if not_aug_transform is None:
                def not_aug_transform(x): return x
            ret_tuple = (apply_transform(selected_samples, transform=not_aug_transform).to(target_device),)
        else:
            ret_tuple = tuple()

        ret_tuple += (apply_transform(selected_samples, transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                selected_attr = attr[choice] if mask_task_out is None else attr[samples_mask][choice]
                ret_tuple += (selected_attr.to(target_device),)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(target_device), ) + ret_tuple

    def get_balanced_data(self, size: int, transform=None, n_classes=-1) -> Tuple:
        """
        Random samples a batch of size items only from n_classes, balancing the samples per class.

        Args:
            size: the number of requested items
            transform: the transformation to be applied (data augmentation)
            n_classes: the number of classes to sample from

        Returns:
            a tuple containing the requested items.
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        tot_classes, class_counts = torch.unique(self.labels[:self.num_seen_examples], return_counts=True)
        if n_classes == -1:
            n_classes = len(tot_classes)

        finished = False
        selected = tot_classes
        while not finished:
            n_classes = min(n_classes, len(selected))
            size_per_class = torch.full([n_classes], size // n_classes)
            size_per_class[:size % n_classes] += 1
            selected = tot_classes[class_counts >= size_per_class[0]]
            if n_classes <= len(selected):
                finished = True
            if len(selected) == 0:
                logging.error('No class has enough examples')
                return self.get_data(size, transform=transform)

        selected = selected[torch.randperm(len(selected))[:n_classes]]

        choice = []
        for i, id_class in enumerate(selected):
            choice += np.random.choice(torch.where(self.labels[:self.num_seen_examples] == id_class)[0].cpu(),
                                       size=size_per_class[i].item(),
                                       replace=False).tolist()
        choice = np.array(choice)

        if transform is None:
            def transform(x): return x
        # ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        ret_tuple = (apply_transform(self.examples[choice], transform=transform).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None, device=None) -> Tuple:
        """
        Returns the data by the given index.

        Args:
            index: the index of the item
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple containing the requested items. The returned items depend on the attributes stored in the buffer from previous calls to `add_data`.
        """
        target_device = self.device if device is None else device

        if transform is None:
            def transform(x): return x
        ret_tuple = (apply_transform(self.examples[indexes], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(target_device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None, device=None) -> Tuple:
        """
        Return all the items in the memory buffer.

        Args:
            transform: the transformation to be applied (data augmentation)

        Returns:
            a tuple with all the items in the memory buffer
        """
        target_device = self.device if device is None else device
        if transform is None:
            ret_tuple = (self.examples[:len(self)].to(target_device),)
        else:
            ret_tuple = (apply_transform(self.examples[:len(self)], transform=transform).to(target_device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)[:len(self)].to(target_device)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0

    def __iter__(self):
        """
        Initializes and returns a iterator object for the buffer.
        """
        self._it_index = 0

        return self

    def __next__(self):
        """
        Returns the next item in the buffer.
        """
        if self._it_index >= self.__len__():
            raise StopIteration
        return self.__getitem__(self._it_index, transform=self._dl_transform)

    def get_dataloader(self, args: Namespace, batch_size: int, shuffle=False, drop_last=False, transform=None, sampler=None) -> torch.utils.data.DataLoader:
        """
        Return a DataLoader for the buffer.

        Args:
            args: the arguments from the CLI
            batch_size: the batch size
            shuffle: if True, shuffle the data
            drop_last: if True, drop the last incomplete batch
            transform: the transformation to be applied (data augmentation)
            sampler: the sampler to be used

        Returns:
            DataLoader
        """
        self._dl_transform = transform
        self._it_index = 0

        return create_seeded_dataloader(args, self, batch_size=batch_size,
                                        shuffle=shuffle, drop_last=drop_last,
                                        sampler=sampler, num_workers=0,
                                        non_verbose=True)

    def __getitem__(self, index, transform=None):
        """
        Returns the item in the buffer at the given index.

        Args:
            index: the index of the item
            transform: (optional) a transformation to be applied

        Returns:
            a tuple containing the requested items. The returned items depend on the attributes stored in the buffer from previous calls to `add_data`.
        """
        data = self.get_data(size=1, transform=transform if self._dl_transform is None or transform is not None else self._dl_transform, force_indexes=[index])

        return [d.squeeze(0) for d in data]


def _select_by_angle(feats: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """
    从一组特征 feats (N, D) 中选择 k 个样本：
      - mode='mid'   : 余弦相似度最接近中位数（中位角）；
      - mode='small' : 余弦相似度最大（角度最小，最贴近均值方向）；
      - mode='big'   : 余弦相似度最小（角度最大，最背离均值方向）。

    返回：选中样本的索引 (k,)
    """
    assert feats.ndim == 2 and feats.size(0) > 0
    N = feats.size(0)
    k = min(k, N)

    # L2 归一化以计算余弦
    nf = F.normalize(feats, dim=1)
    mean_feat = F.normalize(feats.mean(0, keepdim=True), dim=1)

    cos_sim = F.cosine_similarity(nf, mean_feat, dim=1)  # (N,)

    if mode == 'small':
        # 角度最小 -> cos 最大，取 Top-k 大
        order = torch.argsort(cos_sim, descending=True)
        return order[:k]

    if mode == 'big':
        # 角度最大 -> cos 最小，取 Top-k 小
        order = torch.argsort(cos_sim, descending=False)
        return order[:k]

    # mode == 'mid'：与中位数最接近
    median_val = cos_sim.median()
    distances = (cos_sim - median_val).abs()             # (N,)
    order = torch.argsort(distances, descending=False)
    return order[:k]

@torch.no_grad()
def _ipm_select(feats: torch.Tensor, k: int) -> torch.Tensor:
    """
    IPM 选择：从特征矩阵 feats ∈ R^{N×D}（一行一个样本向量）里选 k 个代表样本的索引。
    步骤：
      1) 行向量 L2 归一化得到 A；
      2) 重复 k 次：
         a) 取 A 的第一右奇异向量 v；
         b) 选择与 v 相关性 |A v| 最大的行索引 m；
         c) 把 A 右乘 (I - a_hat a_hat^T) 做空域投影（a_hat 为被选行向量单位化）。
    复杂度 ~ O(k N D)（每轮只用一次 top-1 SVD）。
    """
    assert feats.ndim == 2 and feats.size(0) > 0
    N, D = feats.shape
    k = int(min(k, N))

    # 行归一化（与论文一致：在单位球面上比较夹角/相关性）
    A = F.normalize(feats, dim=1)  # (N, D)

    # 保存“原始行索引”映射（虽然 IPM 不需要删行，但保留便于清晰）
    selected = []

    # 右侧投影的单位阵，后续用于 A @ (I - a a^T)
    I = torch.eye(D, device=A.device, dtype=A.dtype)

    for _ in range(k):
        # 仅需第一右奇异向量
        # torch.linalg.svd 返回 U, S, Vh，其中 Vh 形状 (D, D)，其首行是第一右奇异向量的转置
        # 当 A 退化为零矩阵（数值上可能出现），则提前结束
        if torch.allclose(A, torch.zeros_like(A)):
            break
        try:
            # full_matrices=False 更高效；svd 在大 batch 上更稳
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        except RuntimeError:
            # 数值不稳定兜底：切到 CPU 再来一次
            U, S, Vh = torch.linalg.svd(A.cpu(), full_matrices=False)
            Vh = Vh.to(A.device)
        v = Vh[0]  # (D,)

        # 相关性按 |A v| 最大
        corr = torch.abs(A @ v)  # (N,)
        m = int(torch.argmax(corr).item())
        selected.append(m)

        # 右投影：A ← A (I - a a^T)，其中 a 为当前选择行的单位向量
        a = A[m]               # 已是单位向量
        P = I - torch.outer(a, a)
        A = A @ P

    # 返回“选中的原始行索引”
    if len(selected) == 0:
        # 安全兜底：退回随机 1 个
        return torch.tensor([0], device=feats.device, dtype=torch.long)
    return torch.tensor(selected, device=feats.device, dtype=torch.long)

# @torch.no_grad()
# def fill_buffer(buffer: Buffer, dataset: 'ContinualDataset', t_idx: int, net: 'MammothBackbone' = None, use_herding=False,
#                 required_attributes: List[str] = None, normalize_features=False, extend_equalize_buffer=False) -> None:
#     """
#     Adds examples from the current task to the memory buffer.
#     Supports images, labels, task_labels, and logits.
#
#     Args:
#         buffer: the memory buffer
#         dataset: the dataset from which take the examples
#         t_idx: the task index
#         net: (optional) the model instance. Used if logits are in buffer. If provided, adds logits.
#         use_herding: (optional) if True, uses herding strategy. Otherwise, random sampling.
#         required_attributes: (optional) the attributes to be added to the buffer. If None and buffer is empty, adds only examples and labels.
#         normalize_features: (optional) if True, normalizes the features before adding them to the buffer
#         extend_equalize_buffer: (optional) if True, extends the buffer to equalize the number of samples per class for all classes, even if that means exceeding the buffer size defined at initialization
#     """
#     if net is not None:
#         mode = net.training
#         net.eval()
#     else:
#         assert not use_herding, "Herding strategy requires a model instance"
#
#     device = net.device if net is not None else get_device()
#
#     n_seen_classes = dataset.N_CLASSES_PER_TASK * (t_idx + 1) if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
#         sum(dataset.N_CLASSES_PER_TASK[:t_idx + 1])
#     n_past_classes = dataset.N_CLASSES_PER_TASK * t_idx if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
#         sum(dataset.N_CLASSES_PER_TASK[:t_idx])
#
#     mask = dataset.train_loader.dataset.targets >= n_past_classes
#     dataset.train_loader.dataset.targets = dataset.train_loader.dataset.targets[mask]
#     dataset.train_loader.dataset.data = dataset.train_loader.dataset.data[mask]
#
#     buffer.buffer_size = dataset.args.buffer_size  # reset initial buffer size
#
#     if extend_equalize_buffer:
#         samples_per_class = np.ceil(buffer.buffer_size / n_seen_classes).astype(int)
#         new_bufsize = int(n_seen_classes * samples_per_class)
#         if new_bufsize != buffer.buffer_size:
#             logging.info(f'Buffer size has been changed to: {new_bufsize}')
#         buffer.buffer_size = new_bufsize
#     else:
#         samples_per_class = buffer.buffer_size // n_seen_classes
#
#     # Check for requirs attributes
#     required_attributes = required_attributes or ['examples', 'labels']
#     assert all([attr in buffer.used_attributes for attr in required_attributes]) or len(buffer) == 0, \
#         "Required attributes not in buffer: {}".format([attr for attr in required_attributes if attr not in buffer.used_attributes])
#
#     if t_idx > 0:
#         # 1) First, subsample prior classes
#         buf_data = buffer.get_all_data()
#         buf_y = buf_data[1]
#
#         buffer.empty()
#         for _y in buf_y.unique():
#             idx = (buf_y == _y)
#             _buf_data_idx = {attr_name: _d[idx][:samples_per_class] for attr_name, _d in zip(required_attributes, buf_data)}
#             buffer.add_data(**_buf_data_idx)
#
#     # 2) Then, fill with current tasks
#     loader = dataset.train_loader
#     norm_trans = dataset.get_normalization_transform()
#     if norm_trans is None:
#         def norm_trans(x): return x
#
#     if 'logits' in buffer.used_attributes:
#         assert net is not None, "Logits in buffer require a model instance"
#
#     # 2.1 Extract all features
#     a_x, a_y, a_f, a_l = [], [], [], []
#     for data in loader:
#         x, y, not_norm_x = data[0], data[1], data[2]
#         if not x.size(0):
#             continue
#         a_x.append(not_norm_x.cpu())
#         a_y.append(y.cpu())
#
#         if net is not None:
#             feats = net(norm_trans(not_norm_x.to(device)), returnt='features')
#             outs = net.classifier(feats)
#             if normalize_features:
#                 feats = feats / feats.norm(dim=1, keepdim=True)
#
#             a_f.append(feats.cpu())
#             a_l.append(torch.sigmoid(outs).cpu())
#     a_x, a_y = torch.cat(a_x), torch.cat(a_y)
#     if net is not None:
#         a_f, a_l = torch.cat(a_f), torch.cat(a_l)
#
#     # 2.2 Compute class means
#     for _y in a_y.unique():
#         idx = (a_y == _y)
#         _x, _y = a_x[idx], a_y[idx]
#
#         if use_herding:
#             _l = a_l[idx]
#             feats = a_f[idx]
#             mean_feat = feats.mean(0, keepdim=True)
#
#             running_sum = torch.zeros_like(mean_feat)
#             i = 0
#             while i < samples_per_class and i < feats.shape[0]:
#                 cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)
#
#                 idx_min = cost.argmin().item()
#
#                 buffer.add_data(
#                     examples=_x[idx_min:idx_min + 1].to(device),
#                     labels=_y[idx_min:idx_min + 1].to(device),
#                     logits=_l[idx_min:idx_min + 1].to(device) if 'logits' in required_attributes else None,
#                     task_labels=torch.ones(len(_x[idx_min:idx_min + 1])).to(device) * t_idx if 'task_labels' in required_attributes else None
#
#                 )
#
#                 running_sum += feats[idx_min:idx_min + 1]
#                 feats[idx_min] = feats[idx_min] + 1e6
#                 i += 1
#         else:
#             idx = torch.randperm(len(_x))[:samples_per_class]
#
#             buffer.add_data(
#                 examples=_x[idx].to(device),
#                 labels=_y[idx].to(device),
#                 logits=_l[idx].to(device) if 'logits' in required_attributes else None,
#                 task_labels=torch.ones(len(_x[idx])).to(device) * t_idx if 'task_labels' in required_attributes else None
#             )
#
#     assert len(buffer.examples) <= buffer.buffer_size, f"buffer overflowed its maximum size: {len(buffer)} > {buffer.buffer_size}"
#     assert buffer.num_seen_examples <= buffer.buffer_size, f"buffer has been overfilled, there is probably an error: {buffer.num_seen_examples} > {buffer.buffer_size}"
#
#     if net is not None:
#         net.train(mode)


# ========== 修改版 fill_buffer：只新增 angle_mode 分支，其余保持一致 ==========
@torch.no_grad()
def fill_buffer(buffer: Buffer, dataset: 'ContinualDataset', t_idx: int, net: 'MammothBackbone' = None, use_herding=False,
                required_attributes: List[str] = None, normalize_features=False, extend_equalize_buffer=False,
                angle_mode: str = None) -> None:
    """
    Adds examples from the current task to the memory buffer.
    Supports images, labels, task_labels, and logits.

    Args:
        buffer: the memory buffer
        dataset: the dataset from which take the examples
        t_idx: the task index
        net: (optional) the model instance. Used if logits are in buffer. If provided, adds logits.
        use_herding: (optional) if True, uses herding strategy. Otherwise, random sampling.
        required_attributes: (optional) the attributes to be added to the buffer. If None and buffer is empty, adds only examples and labels.
        normalize_features: (optional) if True, normalizes the features before adding them to the buffer
        extend_equalize_buffer: (optional) if True, extends the buffer to equalize the number of samples per class for all classes, even if that means exceeding the buffer size defined at initialization
        angle_mode: (optional) one of {None,'mid','small','big'}; 若非 None，则启用对应角度策略（需 net）
    """
    if net is not None:
        mode = net.training
        net.eval()
    else:
        # herding 与 angle 策略都需要特征，因此必须有 net
        assert not use_herding and (angle_mode in (None,)), \
            "Herding/angle-based strategies require a model (net)."

    device = net.device if net is not None else get_device()

    n_seen_classes = dataset.N_CLASSES_PER_TASK * (t_idx + 1) if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
        sum(dataset.N_CLASSES_PER_TASK[:t_idx + 1])
    n_past_classes = dataset.N_CLASSES_PER_TASK * t_idx if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
        sum(dataset.N_CLASSES_PER_TASK[:t_idx])

    # 仅保留当前任务样本
    mask = dataset.train_loader.dataset.targets >= n_past_classes
    dataset.train_loader.dataset.targets = dataset.train_loader.dataset.targets[mask]
    dataset.train_loader.dataset.data = dataset.train_loader.dataset.data[mask]

    buffer.buffer_size = dataset.args.buffer_size  # reset initial buffer size

    if extend_equalize_buffer:
        samples_per_class = np.ceil(buffer.buffer_size / n_seen_classes).astype(int)
        new_bufsize = int(n_seen_classes * samples_per_class)
        if new_bufsize != buffer.buffer_size:
            logging.info(f'Buffer size has been changed to: {new_bufsize}')
        buffer.buffer_size = new_bufsize
    else:
        samples_per_class = buffer.buffer_size // n_seen_classes

    # Check for requirs attributes
    required_attributes = required_attributes or ['examples', 'labels']
    assert all([attr in buffer.used_attributes for attr in required_attributes]) or len(buffer) == 0, \
        "Required attributes not in buffer: {}".format([attr for attr in required_attributes if attr not in buffer.used_attributes])

    if t_idx > 0:
        # 1) First, subsample prior classes
        buf_data = buffer.get_all_data()
        buf_y = buf_data[1]

        buffer.empty()
        for _y in buf_y.unique():
            idx = (buf_y == _y)
            _buf_data_idx = {attr_name: _d[idx][:samples_per_class] for attr_name, _d in zip(required_attributes, buf_data)}
            buffer.add_data(**_buf_data_idx)

    # 2) Then, fill with current tasks
    loader = dataset.train_loader
    norm_trans = dataset.get_normalization_transform()
    if norm_trans is None:
        def norm_trans(x): return x

    if 'logits' in buffer.used_attributes:
        assert net is not None, "Logits in buffer require a model instance"

    # 2.1 Extract all features
    a_x, a_y, a_f, a_l = [], [], [], []
    for data in loader:
        x, y, not_norm_x = data[0], data[1], data[2]
        if not x.size(0):
            continue
        a_x.append(not_norm_x.cpu())
        a_y.append(y.cpu())

        if net is not None:
            feats = net(norm_trans(not_norm_x.to(device)), returnt='features')
            outs = net.classifier(feats)
            if normalize_features:
                feats = feats / feats.norm(dim=1, keepdim=True)

            a_f.append(feats.cpu())
            a_l.append(torch.sigmoid(outs).cpu())
    a_x, a_y = torch.cat(a_x), torch.cat(a_y)
    if net is not None:
        a_f, a_l = torch.cat(a_f), torch.cat(a_l)

    # 2.2 Compute class means & select
    use_angle = angle_mode in {'mid', 'small', 'big', 'ipm'}

    for _y in a_y.unique():
        idx = (a_y == _y)
        _x, _y_lbl = a_x[idx], a_y[idx]

        if use_herding:
            # —— 原版 herding（保持一致）——
            _l = a_l[idx]
            feats = a_f[idx]
            mean_feat = feats.mean(0, keepdim=True)

            running_sum = torch.zeros_like(mean_feat)
            i = 0
            while i < samples_per_class and i < feats.shape[0]:
                cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)
                idx_min = cost.argmin().item()

                buffer.add_data(
                    examples=_x[idx_min:idx_min + 1].to(device),
                    labels=_y_lbl[idx_min:idx_min + 1].to(device),
                    logits=_l[idx_min:idx_min + 1].to(device) if 'logits' in required_attributes else None,
                    task_labels=torch.ones(len(_x[idx_min:idx_min + 1])).to(device) * t_idx if 'task_labels' in required_attributes else None
                )

                running_sum += feats[idx_min:idx_min + 1]
                feats[idx_min] = feats[idx_min] + 1e6
                i += 1

        elif use_angle:
            # —— 新增：角度策略（small/mid/big）——
            assert net is not None, "Angle-based strategies require features; please pass `net`."
            feats = a_f[idx].to(device)
            # 若未全局归一化，为余弦计算再做一次 L2 归一化（不改变上游 feats 存储）
            # if not normalize_features:
            #     feats = F.normalize(feats, dim=1)
            if angle_mode == 'ipm':
                # —— 这里走 IPM：类内按 IPM 选出 samples_per_class 个代表 ——
                pick = _ipm_select(feats, samples_per_class)  # (Kc,)
            else:
                # —— 原有 small/mid/big ——
                # _select_by_angle 会内部做行归一化，不必重复
                pick = _select_by_angle(feats, samples_per_class, angle_mode)

            # pick = _select_by_angle(feats, samples_per_class, angle_mode)
            pick_cpu = pick.detach().cpu()

            _x_sel = _x[pick_cpu]
            _y_sel = _y_lbl[pick_cpu]
            _l_sel = a_l[idx][pick_cpu] if net is not None and 'logits' in required_attributes else None

            buffer.add_data(
                examples=_x_sel.to(device),
                labels=_y_sel.to(device),
                logits=_l_sel.to(device) if _l_sel is not None else None,
                task_labels=torch.ones(len(_x_sel)).to(device) * t_idx if 'task_labels' in required_attributes else None
            )

        else:
            # —— 随机 ——
            sel = torch.randperm(len(_x))[:samples_per_class].cpu()
            _l_cur = a_l[idx][sel] if net is not None and 'logits' in required_attributes else None
            buffer.add_data(
                examples=_x[sel].to(device),
                labels=_y_lbl[sel].to(device),
                logits=_l_cur.to(device) if _l_cur is not None else None,
                task_labels=torch.ones(len(sel)).to(device) * t_idx if 'task_labels' in required_attributes else None
            )

    assert len(buffer.examples) <= buffer.buffer_size, f"buffer overflowed its maximum size: {len(buffer)} > {buffer.buffer_size}"
    assert buffer.num_seen_examples <= buffer.buffer_size, f"buffer has been overfilled, there is probably an error: {buffer.num_seen_examples} > {buffer.buffer_size}"

    if net is not None:
        net.train(mode)

# import torch


# def _select_by_angle(feats: torch.Tensor, k: int, mode: str) -> torch.Tensor:
#     """
#     从一组特征 feats (N, D) 中选择 k 个样本：
#       - mode='mid'   : 余弦相似度最接近中位数（中位角）；
#       - mode='small' : 余弦相似度最大（角度最小，最贴近均值方向）；
#       - mode='big'   : 余弦相似度最小（角度最大，最背离均值方向）。
#
#     返回：选中样本的索引 (k,)
#     """
#     assert feats.ndim == 2 and feats.size(0) > 0
#     N = feats.size(0)
#     k = min(k, N)
#
#     # L2 归一化以计算余弦
#     nf = F.normalize(feats, dim=1)
#     mean_feat = F.normalize(nf.mean(0, keepdim=True), dim=1)
#
#     cos_sim = F.cosine_similarity(nf, mean_feat, dim=1)  # (N,)
#
#     if mode == 'small':
#         # 角度最小 -> cos 最大，取 Top-k 大
#         order = torch.argsort(cos_sim, descending=True)
#         return order[:k]
#
#     if mode == 'big':
#         # 角度最大 -> cos 最小，取 Top-k 小
#         order = torch.argsort(cos_sim, descending=False)
#         return order[:k]
#
#     # mode == 'mid'：与中位数最接近
#     median_val = cos_sim.median()
#     distances = (cos_sim - median_val).abs()             # (N,)
#     order = torch.argsort(distances, descending=False)
#     return order[:k]


# from typing import List
# import logging
# import torch
# import torch.nn.functional as F

# @torch.no_grad()
# def fill_buffer_new(buffer: 'Buffer',
#                 dataset: 'ContinualDataset',
#                 t_idx: int,
#                 net: 'MammothBackbone' = None,
#                 use_herding: bool = False,
#                 required_attributes: List[str] = None,
#                 normalize_features: bool = False,
#                 extend_equalize_buffer: bool = False,
#                 angle_mode: str = None  # 新增：None | 'mid' | 'small' | 'big'
#                 ) -> None:
#     """
#     - use_herding=True 时使用 herding（需 net）。
#     - angle_mode in {'mid','small','big'} 时启用角度类策略（需 net）。
#     - 否则走随机分支。
#     """
#     if net is not None:
#         mode = net.training
#         net.eval()
#     else:
#         # herding 或 angle 策略均需要特征，因此必须有 net
#         assert not use_herding and (angle_mode in (None,)), \
#             "Herding/angle-based strategies require a model (net)."
#
#     device = net.device if net is not None else get_device()
#
#     n_seen_classes = dataset.N_CLASSES_PER_TASK * (t_idx + 1) if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
#         sum(dataset.N_CLASSES_PER_TASK[:t_idx + 1])
#     n_past_classes = dataset.N_CLASSES_PER_TASK * t_idx if isinstance(dataset.N_CLASSES_PER_TASK, int) else \
#         sum(dataset.N_CLASSES_PER_TASK[:t_idx])
#
#     # 仅保留当前任务样本
#     mask = dataset.train_loader.dataset.targets >= n_past_classes
#     dataset.train_loader.dataset.targets = dataset.train_loader.dataset.targets[mask]
#     dataset.train_loader.dataset.data = dataset.train_loader.dataset.data[mask]
#
#     # 重置/扩展容量并计算每类配额
#     buffer.buffer_size = dataset.args.buffer_size
#     if extend_equalize_buffer:
#         samples_per_class = int(torch.ceil(torch.tensor(buffer.buffer_size / n_seen_classes)).item())
#         new_bufsize = int(n_seen_classes * samples_per_class)
#         if new_bufsize != buffer.buffer_size:
#             logging.info(f'Buffer size has been changed to: {new_bufsize}')
#         buffer.buffer_size = new_bufsize
#     else:
#         samples_per_class = buffer.buffer_size // n_seen_classes
#
#     # 校验需要的属性
#     required_attributes = required_attributes or ['examples', 'labels']
#     assert all([attr in buffer.used_attributes for attr in required_attributes]) or len(buffer) == 0, \
#         "Required attributes not in buffer: {}".format([attr for attr in required_attributes if attr not in buffer.used_attributes])
#
#     # 旧类均匀下采样以维持类均衡
#     if t_idx > 0 and len(buffer) > 0:
#         buf_data = buffer.get_all_data()
#         buf_y = buf_data[1]
#         buffer.empty()
#         for _y in buf_y.unique():
#             idx = (buf_y == _y)
#             _buf_data_idx = {attr_name: _d[idx][:samples_per_class] for attr_name, _d in zip(required_attributes, buf_data)}
#             buffer.add_data(**_buf_data_idx)
#
#     # 取得标准化变换、准备特征与 logits
#     loader = dataset.train_loader
#     norm_trans = dataset.get_normalization_transform()
#     if norm_trans is None:
#         def norm_trans(x): return x
#
#     if 'logits' in buffer.used_attributes:
#         assert net is not None, "Logits in buffer require a model instance"
#
#     # 扫描当前任务，收集 not_aug_x / y / (feats / logits)
#     a_x, a_y, a_f, a_l = [], [], [], []
#     for data in loader:
#         x, y, not_norm_x = data[0], data[1], data[2]
#         if not x.size(0):
#             continue
#         a_x.append(not_norm_x.cpu())
#         a_y.append(y.cpu())
#
#         if net is not None:
#             feats = net(norm_trans(not_norm_x.to(device)), returnt='features')
#             outs = net.classifier(feats)
#             if normalize_features:
#                 feats = F.normalize(feats, dim=1)
#             a_f.append(feats.cpu())
#             a_l.append(torch.sigmoid(outs).cpu())
#
#     a_x, a_y = torch.cat(a_x), torch.cat(a_y)
#     if net is not None:
#         a_f, a_l = torch.cat(a_f), torch.cat(a_l)
#
#     # 每类选择：herding / 角度类 / 随机
#     use_angle = (angle_mode in {'mid', 'small', 'big'})
#     for _y in a_y.unique():
#         idx = (a_y == _y)
#         _x, _y_lbl = a_x[idx], a_y[idx]
#
#         if use_herding:
#             # —— 原版 herding ——（与你现有实现一致）
#             _l = a_l[idx] if net is not None and 'logits' in required_attributes else None
#             feats = a_f[idx].clone()
#             mean_feat = feats.mean(0, keepdim=True)
#
#             running_sum = torch.zeros_like(mean_feat)
#             i = 0
#             while i < samples_per_class and i < feats.shape[0]:
#                 cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)
#                 idx_min = cost.argmin().item()
#
#                 buffer.add_data(
#                     examples=_x[idx_min:idx_min + 1].to(device),
#                     labels=_y_lbl[idx_min:idx_min + 1].to(device),
#                     logits=_l[idx_min:idx_min + 1].to(device) if _l is not None else None,
#                     task_labels=(torch.ones(1).to(device) * t_idx) if 'task_labels' in required_attributes else None
#                 )
#
#                 running_sum += feats[idx_min:idx_min + 1]
#                 feats[idx_min] = feats[idx_min] + 1e6
#                 i += 1
#
#         elif use_angle:
#             assert net is not None, "Angle-based strategies require features; please pass `net`."
#             feats = a_f[idx].to(device)
#             # 若未全局归一化，这里为余弦再做一次归一化，保证角度语义
#             if not normalize_features:
#                 feats = F.normalize(feats, dim=1)
#
#             pick = _select_by_angle(feats, samples_per_class, angle_mode)
#             _x_sel = _x[pick.cpu()]
#             _y_sel = _y_lbl[pick.cpu()]
#             _l_sel = a_l[idx][pick.cpu()] if net is not None and 'logits' in required_attributes else None
#
#             buffer.add_data(
#                 examples=_x_sel.to(device),
#                 labels=_y_sel.to(device),
#                 logits=_l_sel.to(device) if _l_sel is not None else None,
#                 task_labels=(torch.ones(len(_x_sel)).to(device) * t_idx) if 'task_labels' in required_attributes else None
#             )
#
#         else:
#             # —— 随机 ——
#             sel = torch.randperm(len(_x))[:samples_per_class].cpu()
#             _l_cur = a_l[idx][sel] if net is not None and 'logits' in required_attributes else None
#             buffer.add_data(
#                 examples=_x[sel].to(device),
#                 labels=_y_lbl[sel].to(device),
#                 logits=_l_cur.to(device) if _l_cur is not None else None,
#                 task_labels=(torch.ones(len(sel)).to(device) * t_idx) if 'task_labels' in required_attributes else None
#             )
#
#     # 安全检查
#     assert len(buffer.examples) <= buffer.buffer_size, f"buffer overflowed its maximum size: {len(buffer)} > {buffer.buffer_size}"
#     assert buffer.num_seen_examples <= buffer.buffer_size, f"buffer has been overfilled, there is probably an error: {buffer.num_seen_examples} > {buffer.buffer_size}"
#
#     if net is not None:
#         net.train(mode)


