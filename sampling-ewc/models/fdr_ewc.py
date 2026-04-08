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
from utils.buffer import Buffer


class Fdr(ContinualModel):
    """Continual learning via Function Distance Regularization (+ Online-EWC)."""
    NAME = 'fdr_ewc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight for FDR.')  # 保留原有 FDR 超参
        # ===== 新增 Online-EWC 超参 =====
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for Online-EWC')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma (decay) for Online-EWC Fisher accumulation')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(Fdr, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.i = 0
        self.soft = nn.Softmax(dim=1)
        self.logsoft = nn.LogSoftmax(dim=1)

        # ===== Online-EWC 状态 =====
        self.checkpoint = None  # θ_old
        self.fish = None        # Fisher 对角的在线累积，形状与 get_params() 一致

    # ===== EWC 正则项 =====
    def _ewc_penalty(self):
        if self.checkpoint is None or self.fish is None:
            return torch.tensor(0.0, device=self.device)
        # λ * Σ_i F_i * (θ_i - θ_old_i)^2
        theta = self.net.get_params()
        penalty = self.args.e_lambda * (self.fish * ((theta - self.checkpoint) ** 2)).sum()
        return penalty

    def end_task(self, dataset):
        # ====== 原有 FDR: 维护/均分 buffer ======
        examples_per_task = self.args.buffer_size // self.current_task if self.current_task > 0 else self.args.buffer_size

        if self.current_task > 0:
            buf_x, buf_log, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()

            for ttl in buf_tl.unique():
                idx = (buf_tl == ttl)
                ex, log, tasklab = buf_x[idx], buf_log[idx], buf_tl[idx]
                first = min(ex.shape[0], examples_per_task)
                self.buffer.add_data(
                    examples=ex[:first],
                    logits=log[:first],
                    task_labels=tasklab[:first]
                )
        counter = 0
        with torch.no_grad():
            for i, data in enumerate(dataset.train_loader):
                inputs, not_aug_inputs = data[0], data[2]
                inputs = inputs.to(self.device)
                not_aug_inputs = not_aug_inputs.to(self.device)
                outputs = self.net(inputs)
                if examples_per_task - counter < 0:
                    break
                self.buffer.add_data(
                    examples=not_aug_inputs[:(examples_per_task - counter)],
                    logits=outputs.data[:(examples_per_task - counter)],
                    task_labels=(torch.ones(self.args.batch_size, device=self.device) *
                                 self.current_task)[:(examples_per_task - counter)]
                )
                counter += self.args.batch_size

        # # ---- 1) 备份训练/BN状态，并冻结 BN 统计 ----
        # was_training = self.net.training
        # # 收集 BN 层以备恢复
        # bn_modules = []
        # bn_states = []
        # for m in self.net.modules():
        #     if isinstance(m, nn.modules.batchnorm._BatchNorm):
        #         bn_modules.append(m)
        #         # 这三个 buffer/计数需要完整还原
        #         rm = m.running_mean.clone() if m.running_mean is not None else None
        #         rv = m.running_var.clone() if m.running_var is not None else None
        #         nbt = m.num_batches_tracked.clone() if hasattr(m, "num_batches_tracked") else None
        #         bn_states.append((rm, rv, nbt))
        #
        # # 关键：切到 eval，防止 BN 更新 running 统计（仍可做反向传播以取梯度）
        # self.net.eval()

        # ====== 新增 Online-EWC: 估计 Fisher 并在线累积 ======
        fish = torch.zeros_like(self.net.get_params())

        # 逐样本近似 E[ (∇_θ log p_θ(y|x))^2 ]
        for j, data in enumerate(dataset.train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                # log p(y|x) = - nll_loss(logsoft(output), y)
                log_py_x = -F.nll_loss(self.logsoft(output), lab.unsqueeze(0), reduction='none')
                # 可选再加权：p(y|x) = exp(log p)
                weight = torch.mean(torch.exp(log_py_x.detach().clone()))
                loss = torch.mean(log_py_x)
                loss.backward()
                # get_grads() 应返回与 get_params() 同形的一维向量
                fish += weight * (self.net.get_grads() ** 2)

        num_samples = len(dataset.train_loader) * self.args.batch_size if len(dataset.train_loader) > 0 else 1
        fish /= max(1, num_samples)

        if self.fish is None:
            self.fish = fish
        else:
            # 在线衰减累积
            self.fish *= self.args.gamma
            self.fish += fish

        # 更新 θ_old
        self.checkpoint = self.net.get_params().data.clone()

        # # ---- 3) 恢复 BN running 统计与训练/评估模式 ----
        # for m, (rm, rv, nbt) in zip(bn_modules, bn_states):
        #     if rm is not None:  m.running_mean.copy_(rm)
        #     if rv is not None:  m.running_var.copy_(rv)
        #     if nbt is not None and hasattr(m, "num_batches_tracked"):
        #         m.num_batches_tracked.copy_(nbt)
        #
        # if was_training:
        #     self.net.train()
        # else:
        #     self.net.eval()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.i += 1

        # ===== 第一次优化：当前批次的监督 CE +（可选）EWC 正则 =====
        self.opt.zero_grad()
        outputs = self.net(inputs)
        ce_loss = self.loss(outputs, labels)

        # 仅添加 EWC；FDR 的 α 惩罚不改动（你原代码中未显式使用 α，这里仍保持不动）
        loss1 = ce_loss
        assert not torch.isnan(loss1)
        loss1.backward()
        self.opt.step()

        # ===== 第二次优化：buffer 上的 FDR distillation +（可选）EWC 正则 =====
        if not self.buffer.is_empty():
            self.opt.zero_grad()
            buf_inputs, buf_logits, _ = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform, device=self.device
            )
            buf_outputs = self.net(buf_inputs)
            fdr_loss = torch.norm(self.soft(buf_outputs) - self.soft(buf_logits), p=2, dim=1).mean()

            loss2 = fdr_loss + self._ewc_penalty()
            assert not torch.isnan(loss2)
            loss2.backward()
            self.opt.step()
            return loss2.item()

        return loss1.item()
