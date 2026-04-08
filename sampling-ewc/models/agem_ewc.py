# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega,
# Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gem import overwrite_grad, store_grad
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    # 将 gxy 投影到与 ger 不冲突的方向（A-GEM）
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


class AGemEWC(ContinualModel):
    """Continual learning via A-GEM + Online-EWC (o-ewc + agem)."""
    NAME = 'agem_ewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)  # --buffer_size, --minibatch_size ...
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for Online-EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='Fisher decay for Online-EWC')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AGemEWC, self).__init__(backbone, loss, args, transform, dataset=dataset)

        # ----- A-GEM / ER -----
        self.buffer = Buffer(self.args.buffer_size)
        self.grad_dims = [param.data.numel() for param in self.parameters()]
        flat_n = int(np.sum(self.grad_dims))
        self.grad_xy = torch.Tensor(flat_n).to(self.device)  # 当前(任务)梯度（展平）
        self.grad_er = torch.Tensor(flat_n).to(self.device)  # buffer 梯度（展平）

        # ----- Online-EWC -----
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None                        # θ_old (展平)
        self.fish = None                              # Fisher 对角（展平）
        self._tmp_grad = torch.zeros_like(self.grad_xy)  # Fisher 估计时的临时展平梯度

    # ========================= EWC helpers =========================

    def _flatten_params(self) -> torch.Tensor:
        # 将当前参数展平为单个向量（不需要梯度）
        flats = []
        for p in self.parameters():
            flats.append(p.data.view(-1))
        return torch.cat(flats).to(self.device)

    def _get_penalty_grads_flat(self):
        """
        返回 EWC 惩罚的展平梯度向量：
        ∂/∂θ [ λ * Σ_i F_i (θ_i - θ_i_old)^2 ] = 2λ F ⊙ (θ - θ_old)
        """
        if self.checkpoint is None or self.fish is None:
            return None
        theta_flat = self._flatten_params()
        return 2.0 * self.args.e_lambda * self.fish * (theta_flat - self.checkpoint)

    @torch.no_grad()
    def _update_snapshot(self):
        self.checkpoint = self._flatten_params().clone()

    def _estimate_fisher_online(self, dataset):
        """
        在当前任务数据上在线估计 Fisher（对角近似，逐样本梯度平方），并按 gamma 衰减累积。
        注意：逐样本反传较慢，如需加速可改成 mini-batch 近似。
        """
        device = self.device
        fish_new = torch.zeros_like(self.grad_xy, device=device)

        total = 0
        self.net.train()
        for _, data in enumerate(dataset.train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            for ex, lab in zip(inputs, labels):
                total += 1
                self.opt.zero_grad(set_to_none=True)
                out = self.net(ex.unsqueeze(0))  # [1, C]

                # 与常见 EWC 做法一致，使用 NLL(真实标签) 的梯度平方近似 Fisher 对角
                nll = F.nll_loss(self.logsoft(out), lab.unsqueeze(0), reduction='mean')
                nll.backward()

                # 将逐样本梯度展平到 _tmp_grad
                store_grad(self.parameters, self._tmp_grad, self.grad_dims)
                fish_new += self._tmp_grad.pow(2)

        if total > 0:
            fish_new /= float(total)

        if self.fish is None:
            self.fish = fish_new
        else:
            # Online 累积：F ← γ F + F_new
            self.fish.mul_(self.args.gamma).add_(fish_new)

        # 更新 θ_old
        self._update_snapshot()

    # ========================= Mammoth hooks =========================

    def end_task(self, dataset):
        """
        每个任务结束时：估计 Fisher 并更新 θ_old（在线 EWC）
        """
        self._estimate_fisher_online(dataset)

        # （保持与原 agem 同步的最小行为）可在此也向 buffer 写入部分该任务样本
        # 原文件仅取一个 batch，这里沿用：
        samples_per_task = max(1, self.args.buffer_size // max(1, dataset.N_TASKS))
        loader = dataset.train_loader
        cur_y, cur_x = next(iter(loader))[1:]
        self.buffer.add_data(
            examples=cur_x.to(self.device),
            labels=cur_y.to(self.device)
        )

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        单步训练：
        1) 用当前 batch 得到原始梯度 g_xy；
        2) 若 buffer 非空：按 A-GEM 用 g_er 对 g_xy 做投影，得到 g_agem（与无 EWC 时完全一致）；
           否则：g_agem = g_xy；
        3) 计算 EWC 惩罚梯度 g_ewc（若已有 Fisher/θ_old），并加到 g_agem 上得到最终梯度；
        4) 覆盖梯度并 step。
        """
        self.zero_grad()

        # -------- 当前 batch 的监督损失与梯度：g_xy --------
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        p = self.net.forward(inputs)
        loss = self.loss(p, labels)
        loss.backward()

        # 保存当前梯度到 grad_xy
        store_grad(self.parameters, self.grad_xy, self.grad_dims)

        # 预先计算 EWC 惩罚梯度（展平向量），但此时不加到 g_xy 上
        pen_flat = self._get_penalty_grads_flat()  # 可能为 None

        # -------- A-GEM：与无 EWC 时相同的投影逻辑，得到 g_agem --------
        if not self.buffer.is_empty():
            # 计算 buffer 梯度 g_er
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            self.net.zero_grad()
            buf_outputs = self.net.forward(buf_inputs)
            penalty = self.loss(buf_outputs, buf_labels)
            penalty.backward()
            store_grad(self.parameters, self.grad_er, self.grad_dims)

            # 原版 A-GEM：若冲突则把 g_xy 投影到与 g_er 不冲突的方向
            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_agem = project(gxy=self.grad_xy, ger=self.grad_er)
            else:
                g_agem = self.grad_xy.clone()
        else:
            # 无 buffer：g_agem 就是 g_xy
            g_agem = self.grad_xy.clone()

        # -------- 在 A-GEM 之后再加 EWC 惩罚梯度 --------
        if pen_flat is not None:
            g_agem.add_(pen_flat)

        # 覆盖梯度并更新
        overwrite_grad(self.parameters, g_agem, self.grad_dims)
        self.opt.step()

        return loss.item()

