# Copyright 2022-present, Lorenzo Bonicelli,
# Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import get_dataset  # noqa: F401 (kept for consistency with other models)
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def dsimplex(num_classes: int = 10):
    """
    生成正单纯形（regular simplex）原型矩阵（单位范数、列间等距），形状为 [num_classes-1, num_classes]。
    """
    def simplex_coordinates2(m: int):
        import numpy as np
        x = np.zeros([m, m + 1])
        for j in range(m):
            x[j, j] = 1.0

        a = (1.0 - np.sqrt(float(1 + m))) / float(m)
        for i in range(m):
            x[i, m] = a

        # 质心归零
        c = np.zeros(m)
        for i in range(m):
            s = 0.0
            for j in range(m + 1):
                s += x[i, j]
            c[i] = s / float(m + 1)
        for j in range(m + 1):
            for i in range(m):
                x[i, j] = x[i, j] - c[i]

        # 列单位化
        s = 0.0
        for i in range(m):
            s += x[i, 0] ** 2
        s = np.sqrt(s)
        for j in range(m + 1):
            for i in range(m):
                x[i, j] = x[i, j] / s
        return x

    feat_dim = num_classes - 1
    ds = simplex_coordinates2(feat_dim)
    return ds


class RPCewc(ContinualModel):
    """Regular Polytope Classifier + Online-EWC (penalty 加到 loss 一次性反传)."""
    NAME = 'rpc_ewc'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # 原 ER 参数
        add_rehearsal_args(parser)
        # === Online-EWC 相关参数 ===
        parser.add_argument('--e_lambda', type=float, default=0.0,
                            help='lambda weight for Online-EWC (0 to disable).')
        parser.add_argument('--gamma', type=float, default=1.0,
                            help='gamma (decay) for Online-EWC fisher accumulation.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(RPCewc, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # 总类别数（已知任务数 × 每任务类别数）
        self.total_classes = self.cpt * self.n_tasks
        # 固定几何分类头：形状 [feat_dim, total_classes]，feat_dim = total_classes - 1
        rpch = torch.from_numpy(dsimplex(self.total_classes)).float()
        self.register_buffer("rpchead", rpch.to(self.device))  # 作为 buffer，不参与训练

        # 懒初始化特征投影层（用于对齐到 feat_dim）
        self._proj = None
        self._feat_dim = self.rpchead.shape[0]  # = total_classes - 1

        # === Online-EWC 状态 ===
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None   # θ_old
        self.fish = None         # Fisher 累积（展平，与 get_params 同形）

    # --------- 特征抽取与对齐 ---------
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        try:
            z = self.net(x, returnt='features')
            return z
        except Exception:
            return self.net(x)

    def _maybe_init_proj(self, in_dim: int):
        if self._proj is None and in_dim != self._feat_dim:
            self._proj = nn.Linear(in_dim, self._feat_dim, bias=False).to(self.device)

    def _align_features(self, z: torch.Tensor) -> torch.Tensor:
        in_dim = z.shape[1]
        if in_dim == self._feat_dim:
            return z
        self._maybe_init_proj(in_dim)
        if self._proj is not None:
            return self._proj(z)
        else:
            return z[:, :self._feat_dim]

    # --------- 前向：统一训练/评测路径 ---------
    def forward(self, x):
        z = self._extract_features(x)                # [N, D_in]
        z = self._align_features(z)                  # [N, feat_dim]
        logits = z @ self.rpchead                    # [N, total_classes]
        return logits

    # --------- Online-EWC: penalty 标量 ---------
    def penalty(self):
        """
        λ * Σ F (θ - θ_old)^2
        当 checkpoint 或 Fisher 不存在，或 λ=0 时，返回 0。
        """
        if (self.checkpoint is None) or (self.fish is None) or (self.args.e_lambda == 0.0):
            return torch.tensor(0.0, device=self.device)
        theta = self.net.get_params()
        return self.args.e_lambda * (self.fish * ((theta - self.checkpoint) ** 2)).sum()

    # --------- 任务结束：维护 buffer（原逻辑不变）+ 估计/累积 Fisher（Online-EWC） ---------
    def end_task(self, dataset):
        # ---------------- 原：reduce coreset ----------------
        if self.current_task > 0:
            examples_per_class = self.args.buffer_size // (self.current_task * self.cpt)
            buf_x, buf_lab = self.buffer.get_all_data()
            self.buffer.empty()
            for tl in buf_lab.unique():
                idx = tl == buf_lab
                ex, lab = buf_x[idx], buf_lab[idx]
                first = min(ex.shape[0], examples_per_class)
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=lab[:first]
                )

        # ---------------- 原：add new task ----------------
        examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
        if examples_last_task > 0:
            examples_per_class = examples_last_task // self.cpt
            ce = torch.tensor([examples_per_class] * self.cpt).int()
            remain = examples_last_task - (examples_per_class * self.cpt)
            if remain > 0:
                ce[torch.randperm(self.cpt)[:remain]] += 1

            with torch.no_grad():
                for data in dataset.train_loader:
                    labels, not_aug_inputs = data[1], data[2]
                    not_aug_inputs = not_aug_inputs.to(self.device)
                    if all(ce == 0):
                        break

                    flags = torch.zeros(len(labels), dtype=torch.bool)
                    for j in range(len(flags)):
                        cls = int(labels[j].item()) % self.cpt
                        if ce[cls] > 0:
                            flags[j] = True
                            ce[cls] -= 1

                    if flags.any():
                        self.buffer.add_data(
                            examples=not_aug_inputs[flags],
                            labels=labels[flags]
                        )

        # ---------------- 新增：Online-EWC 的 Fisher 估计与累积 ----------------
        if self.args.e_lambda != 0.0:
            # ---- 1) 备份训练/BN状态，并冻结 BN 统计 ----
            was_training = self.net.training
            # 收集 BN 层以备恢复
            bn_modules = []
            bn_states = []
            for m in self.net.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    bn_modules.append(m)
                    # 这三个 buffer/计数需要完整还原
                    rm = m.running_mean.clone() if m.running_mean is not None else None
                    rv = m.running_var.clone() if m.running_var is not None else None
                    nbt = m.num_batches_tracked.clone() if hasattr(m, "num_batches_tracked") else None
                    bn_states.append((rm, rv, nbt))

            # 关键：切到 eval，防止 BN 更新 running 统计（仍可做反向传播以取梯度）
            self.net.eval()

            fish = torch.zeros_like(self.net.get_params())

            # 与 Mammoth 的 EwcOn 实现一致的近似：逐样本梯度 + exp(cond prob) 加权
            for _, data in enumerate(dataset.train_loader):
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                for ex, lab in zip(inputs, labels):
                    self.opt.zero_grad()
                    output = self.net(ex.unsqueeze(0))
                    # 负对数似然
                    loss = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                    exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                    loss = torch.mean(loss)
                    loss.backward()
                    fish += exp_cond_prob * self.net.get_grads() ** 2

            # 按样本数归一化
            fish /= (len(dataset.train_loader) * self.args.batch_size)

            if self.fish is None:
                self.fish = fish
            else:
                # 在线指数衰减累积
                self.fish *= self.args.gamma
                self.fish += fish

            # 保存参数快照 θ_old
            self.checkpoint = self.net.get_params().data.clone()

            # ---- 3) 恢复 BN running 统计与训练/评估模式 ----
            for m, (rm, rv, nbt) in zip(bn_modules, bn_states):
                if rm is not None:  m.running_mean.copy_(rm)
                if rv is not None:  m.running_var.copy_(rv)
                if nbt is not None and hasattr(m, "num_batches_tracked"):
                    m.num_batches_tracked.copy_(nbt)

            if was_training:
                self.net.train()
            else:
                self.net.eval()

    # --------- 训练步：一次性反传（CE + EWC penalty） ---------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        outputs = self(inputs)

        # CE
        ce_loss = self.loss(outputs, labels)

        # EWC penalty（若无 checkpoint / fisher / λ=0，则返回 0）
        ewc_penalty = self.penalty()

        # 一次性反传
        loss = ce_loss + ewc_penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return float(loss.item())
