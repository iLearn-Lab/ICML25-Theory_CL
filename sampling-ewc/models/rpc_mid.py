# Copyright 2022-present, Lorenzo Bonicelli,
# Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from datasets import get_dataset  # noqa: F401 (kept for consistency with other models)
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, fill_buffer  # <- 加入 fill_buffer


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


class RPCMid(ContinualModel):
    """Regular Polytope Classifier."""
    NAME = 'rpc_mid'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(RPCMid, self).__init__(backbone, loss, args, transform, dataset=dataset)
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

    # --------- 任务结束：维护 buffer（旧任务均衡保留 + 当前任务 mid-angle 选样） ---------
    def end_task(self, dataset):
        # 1) 旧任务样本：按类均衡下采样（保持与原版一致）
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

        # 2) 当前任务样本：用 mid-angle 从本任务训练集选样加入 buffer（替换原先随机/顺序配额）
        self.net.eval()
        with torch.no_grad():
            fill_buffer(
                buffer=self.buffer,
                dataset=dataset,
                t_idx=self.current_task,
                net=self.net,
                use_herding=False,          # 不用 herding
                angle_mode='mid',           # 使用 mid-angle 取样
                normalize_features=False,   # 由 fill_buffer 内部进行 L2 归一化后再算余弦（按你们实现）
                extend_equalize_buffer=False
            )
        self.net.train()

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
