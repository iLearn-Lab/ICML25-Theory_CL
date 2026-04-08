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


class RPC(ContinualModel):
    """Regular Polytope Classifier."""
    NAME = 'rpc'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(RPC, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # 总类别数（已知任务数 × 每任务类别数）
        self.total_classes = self.cpt * self.n_tasks
        # 固定几何分类头：形状 [feat_dim, total_classes]，feat_dim = total_classes - 1
        rpch = torch.from_numpy(dsimplex(self.total_classes)).float()
        self.register_buffer("rpchead", rpch.to(self.device))  # 作为 buffer，不参与训练

        # 将在第一次前向时“懒初始化”特征投影层，使得特征维度对齐 rpchead 行数
        self._proj = None  # nn.Linear(backbone_dim, feat_dim, bias=False) 将在首次 forward 配置
        self._feat_dim = self.rpchead.shape[0]  # = total_classes - 1

    # --------- 特征抽取与对齐 ---------
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        尝试优先拿 backbone 的“特征”，否则拿默认前向输出作为特征。
        """
        # 一些 Mammoth/backbone 支持 returnt='features'
        try:
            z = self.net(x, returnt='features')
            return z
        except Exception:
            pass
        # 回退到普通前向（可能是 logits 或倒数第二层输出，依具体 backbone）
        return self.net(x)

    def _maybe_init_proj(self, in_dim: int):
        """
        按需懒初始化线性投影到 feat_dim，避免“拍脑袋切一维”的脆弱做法。
        """
        if self._proj is None and in_dim != self._feat_dim:
            self._proj = nn.Linear(in_dim, self._feat_dim, bias=False).to(self.device)

    def _align_features(self, z: torch.Tensor) -> torch.Tensor:
        """
        将特征对齐到 rpchead 的行数（feat_dim）。
        - 若维度相等：直接返回
        - 若维度不同：
            * 若已建立投影层：z @ W
            * 未建立且维度更大：安全地取前 feat_dim 维（避免破坏已有训练）
            * 未建立且维度更小：懒初始化投影再映射到更高维
        """
        in_dim = z.shape[1]
        if in_dim == self._feat_dim:
            return z

        # 首次遇到维度不符时，建立或决定策略
        self._maybe_init_proj(in_dim)

        if self._proj is not None:
            return self._proj(z)
        else:
            # 没建投影层，说明 in_dim > feat_dim；用安全切片以保证可微与可跑
            return z[:, :self._feat_dim]

    # --------- 前向：统一训练/评测路径 ---------
    def forward(self, x):
        z = self._extract_features(x)                # [N, D_in]
        z = self._align_features(z)                  # [N, feat_dim]
        logits = z @ self.rpchead                    # [N, total_classes]
        return logits

    # --------- 任务结束：维护 buffer（与你原逻辑一致） ---------
    def end_task(self, dataset):
        # reduce coreset
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

        # add new task
        examples_last_task = self.buffer.buffer_size - self.buffer.num_seen_examples
        if examples_last_task <= 0:
            return

        examples_per_class = examples_last_task // self.cpt
        ce = torch.tensor([examples_per_class] * self.cpt).int()
        # 将多出来的若干个名额随机打散到各类
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

    # --------- 训练步：统一用 forward（= 几何头） ---------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device)
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        # 关键：用 self(...) 前向，确保训练与评测一致（特征 → rpchead）
        outputs = self(inputs)
        loss = self.loss(outputs, labels)

        loss.backward()
        self.opt.step()
        return float(loss.item())

