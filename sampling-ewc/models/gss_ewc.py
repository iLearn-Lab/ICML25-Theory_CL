# Copyright 2020-present, Pietro Buzzega, Matteo Boschini,
# Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.gss_buffer import Buffer as Buffer


class GssEwc(ContinualModel):
    """Gradient based sample selection for online continual learning + Online-EWC."""
    NAME = 'gss_ewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # 原有 GSS/ER 参数
        add_rehearsal_args(parser)
        parser.add_argument('--batch_num', type=int, default=1,
                            help='Number of batches extracted from the buffer.')
        parser.add_argument('--gss_minibatch_size', type=int, default=None,
                            help='The batch size of the gradient comparison.')

        # === Online-EWC 相关新增参数 ===
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for Online-EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for Online-EWC (decay for accumulated Fisher)')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(GssEwc, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(
            self.args.buffer_size,
            self.device,
            self.args.gss_minibatch_size if self.args.gss_minibatch_size is not None
            else self.args.minibatch_size,
            self
        )
        self.alj_nepochs = self.args.batch_num

        # === Online-EWC 状态 ===
        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None   # θ_old
        self.fish = None         # 累积 Fisher，与 get_params 同形

    # -------- GSS 原有梯度获取（供 gss_buffer 使用）--------
    def get_grads(self, inputs, labels):
        self.net.eval()
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        grads = self.net.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()
        if len(grads.shape) == 1:
            grads = grads.unsqueeze(0)
        return grads

    # -------- Online-EWC：正则项 --------
    def penalty(self):
        if self.checkpoint is None or self.fish is None:
            return torch.tensor(0.0, device=self.device)
        # λ * Σ_i F_i * (θ_i - θ*_i)^2
        return self.args.e_lambda * (self.fish * (self.net.get_params() - self.checkpoint) ** 2).sum()

    # -------- Online-EWC：在任务结束时估计 Fisher 并滚动累积 --------
    def end_task(self, dataset):
        """
        在每个任务结束时被框架调用。
        用 log p(y|x) 的梯度平方近似 Fisher，并以 gamma 衰减做 online 累积。
        """
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

        # 形状与参数向量相同
        fish = torch.zeros_like(self.net.get_params())

        # 逐样本估计 Fisher（与 Mammoth EWC 一致的写法）
        for batch in dataset.train_loader:
            inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                # log p(y|x): 先 LogSoftmax，再 nll_loss 加负号得到 log-likelihood
                logp = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                # 用 p(y|x) 作为稳定化权重（与 Mammoth 实现一致）
                w = torch.mean(torch.exp(logp.detach().clone()))
                # 对 log-likelihood 求梯度
                loss = torch.mean(logp)
                loss.backward()
                # grad^2 累加
                fish += w * (self.net.get_grads() ** 2)

        # 归一化（按样本数均值）
        num_samples = len(dataset.train_loader) * self.args.batch_size
        if num_samples > 0:
            fish /= num_samples

        # Online 累积：旧的乘 gamma 衰减，再加上本任务的
        if self.fish is None:
            self.fish = fish
        else:
            self.fish.mul_(self.args.gamma).add_(fish)

        # 记录旧参数快照 θ_old
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

    # -------- 训练一步：保持 GSS 流程不变，仅把 EWC 正则并入损失 --------
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]
        # GSS 缓存维护（保持不变）
        self.buffer.drop_cache()
        self.buffer.reset_fathom()

        # 在当前 batch 上进行 alj_nepochs 次更新（保持不变）
        for _ in range(self.alj_nepochs):
            self.opt.zero_grad()
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform
                )
                tinputs = torch.cat((inputs, buf_inputs))
                tlabels = torch.cat((labels, buf_labels))
            else:
                tinputs = inputs
                tlabels = labels

            outputs = self.net(tinputs)
            ce_loss = self.loss(outputs, tlabels)

            # === 唯一的改动：把 Online-EWC 正则项并入总损失 ===
            if self.checkpoint is not None and self.fish is not None:
                loss = ce_loss + self.penalty()
            else:
                loss = ce_loss

            assert not torch.isnan(loss)
            loss.backward()
            self.opt.step()

        # 训练后，按 GSS 的策略向 buffer 加入未经增强的当前样本（保持不变）
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()
