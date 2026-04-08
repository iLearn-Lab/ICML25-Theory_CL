# -*- coding: utf-8 -*-
"""
EWC (online) + ER (experience replay) for Mammoth.

- 训练阶段（observe）：
  * 用 ER 将当前 batch 与 buffer 样本拼接训练；
  * 在反传前注入 EWC 惩罚梯度（与 EwcOn 一致的风格）。

- 任务结束（end_task）：
  * 在该任务数据上估计 Fisher（逐样本梯度平方近似）；
  * 按 gamma 衰减累积（Online EWC）；
  * 更新参数快照 checkpoint。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class EwcOnEr(ContinualModel):
    """Continual learning via Online-EWC + Experience Replay."""
    NAME = 'er_ewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        同时加入 ER 的缓冲区参数与 Online-EWC 的超参。
        """
        add_rehearsal_args(parser)  # 提供 --buffer_size, --minibatch_size 等
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for online EWC (Fisher decay)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(EwcOnEr, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # ---- ER ----
        self.buffer = Buffer(self.args.buffer_size)

        # ---- EWC (online) ----
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None   # θ_old
        self.fish = None         # 累积的 Fisher（带 gamma 衰减）

    # ====================== EWC helpers ======================

    def penalty(self):
        """
        可选：EWC 惩罚项（用于日志/调试）。主训练仍采用“注入梯度”方式。
        """
        if self.checkpoint is None or self.fish is None:
            return torch.tensor(0.0, device=self.device)
        return self.args.e_lambda * (self.fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()

    def get_penalty_grads(self):
        """
        d/dθ [ λ * Σ_i F_i (θ_i - θ_i_old)^2 ] = 2λ F ⊙ (θ - θ_old)
        返回展平梯度，配合 self.net.set_grads 使用。
        """
        if self.checkpoint is None or self.fish is None:
            return None
        return self.args.e_lambda * 2.0 * self.fish * (self.net.get_params().data - self.checkpoint)

    def _estimate_fisher_online(self, dataset):
        """
        在当前任务数据上估计 Fisher（逐样本梯度平方），并按 gamma 做在线累积。
        注意：不能在 no_grad 环境中，否则 loss.backward() 会报不需要梯度。
        """
        self.net.train()

        fish = torch.zeros_like(self.net.get_params())

        for _, data in enumerate(dataset.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # 逐样本估计梯度平方
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad(set_to_none=True)
                output = self.net(ex.unsqueeze(0))  # [1, C]

                # 与 EwcOn 一致：先取“对数似然”（负的 nll），再用 exp 缩放
                nll = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')  # shape: [1]
                exp_cond_prob = torch.mean(torch.exp(nll.detach().clone()))
                loss = nll.mean()  # 标量
                loss.backward()

                fish += exp_cond_prob * (self.net.get_grads() ** 2)

        # 归一化（按样本数）
        denom = (len(dataset.train_loader) * self.args.batch_size)
        if denom > 0:
            fish /= float(denom)

        # Online 累积：旧的乘 gamma，再加新的
        if self.fish is None:
            self.fish = fish
        else:
            self.fish.mul_(self.args.gamma).add_(fish)

        # 参数快照 θ_old
        self.checkpoint = self.net.get_params().data.clone()

    # ====================== Mammoth hooks ======================

    def end_task(self, dataset):
        """
        每个任务结束时由框架调用：此处进行 Fisher 估计与快照更新。
        """
        self._estimate_fisher_online(dataset)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        单步训练：
        1) 从 buffer 取样拼接（ER）；
        2) 前向 + 监督损失；
        3) 若已有 Fisher/快照，注入 EWC 梯度；
        4) 反传与更新参数；
        5) 将当前任务的原始样本存入 buffer。
        """
        self.opt.zero_grad(set_to_none=True)

        real_batch_size = inputs.shape[0]
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # ---- ER：从 buffer 取样并拼接 ----
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        # ---- 前向与监督损失 ----
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        if torch.isnan(loss):
            raise RuntimeError("NaN loss encountered.")

        # ---- 注入 EWC 惩罚梯度（若已有 Fisher 与快照）----
        pen_grads = self.get_penalty_grads()
        if pen_grads is not None:
            self.net.set_grads(pen_grads)  # 先放入惩罚梯度，再叠加 task loss 的梯度

        # ---- 反传 + 更新 ----
        loss.backward()
        self.opt.step()

        # ---- 仅将当前任务样本写入 buffer（未增强版本）----
        self.buffer.add_data(
            examples=not_aug_inputs,                   # 非增广原始输入
            labels=labels[:real_batch_size].detach()   # 只保存当前任务这部分的标签
        )

        return loss.item()
