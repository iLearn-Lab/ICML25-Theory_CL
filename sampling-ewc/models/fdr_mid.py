# -*- coding: utf-8 -*-
# Copyright 2020-present,
#   Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, fill_buffer


class FdrMid(ContinualModel):
    """Continual learning via Function Distance Regularization (mid-angle sampling)."""
    NAME = 'fdr_mid'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(FdrMid, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)
        self.i = 0
        self.soft = torch.nn.Softmax(dim=1)
        self.logsoft = torch.nn.LogSoftmax(dim=1)

    def begin_task(self, dataset) -> None:
        self.net.train()

    @torch.no_grad()
    def _readd_old_tasks_with_quota(self, examples_per_task: int) -> None:
        """
        将旧缓冲区按“每任务配额”回写，保持与原 FDR 相同的旧样本保留策略。
        兼容 buffer.get_all_data() 返回三元组或四元组。
        """
        if self.buffer.is_empty():
            return

        all_buf = self.buffer.get_all_data()
        # 允许三元组(examples, logits, task_labels)或四元组(examples, labels, logits, task_labels)
        if len(all_buf) == 3:
            buf_x, buf_log, buf_tl = all_buf
            buf_y = None
        elif len(all_buf) == 4:
            buf_x, buf_y, buf_log, buf_tl = all_buf
        else:
            # 非预期情形：只取我们需要的字段（0/2/3）
            buf_x, buf_log, buf_tl = all_buf[0], all_buf[2], all_buf[3]
            buf_y = all_buf[1] if len(all_buf) > 1 else None

        # 清空后按每任务配额回写
        self.buffer.empty()
        for ttl in buf_tl.unique():
            idx = (buf_tl == ttl)
            ex, log, tl = buf_x[idx], buf_log[idx], buf_tl[idx]
            first = min(ex.shape[0], examples_per_task)
            if buf_y is not None:
                y = buf_y[idx]
                self.buffer.add_data(
                    examples=ex[:first],
                    labels=y[:first],
                    logits=log[:first],
                    task_labels=tl[:first]
                )
            else:
                self.buffer.add_data(
                    examples=ex[:first],
                    logits=log[:first],
                    task_labels=tl[:first]
                )

    def end_task(self, dataset):
        """
        任务结束时：
        1) 先按“每任务配额”保留旧任务样本（保持原 FDR 行为）；
        2) 再用 mid-angle 在“当前任务训练集”上选样写入（examples/labels/logits/task_labels）。
        """
        # 与原实现一致：当前任务>0时，按配额保留旧任务样本
        examples_per_task = self.args.buffer_size // self.current_task if self.current_task > 0 else self.args.buffer_size

        if self.current_task > 0:
            self._readd_old_tasks_with_quota(examples_per_task)

        # —— 用 mid-angle 从“当前任务训练集”中选代表样本并写入 —— #
        # 说明：
        # - 必须包含 labels 才能做类筛选，且需要 logits 供 FDR 的分布距离约束使用
        # - required_attributes 顺序必须与 get_all_data 返回一致：examples, labels, logits, task_labels
        # - normalize_features=False：仅在计算余弦时做临时 L2 归一化，不改动存储特征
        self.net.eval()
        with torch.no_grad():
            fill_buffer(
                buffer=self.buffer,
                dataset=dataset,
                t_idx=self.current_task,
                net=self.net,
                use_herding=False,
                required_attributes=['examples', 'labels', 'logits', 'task_labels'],
                angle_mode='mid',
                normalize_features=False,
                extend_equalize_buffer=False
            )
        self.net.train()

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.i += 1

        # —— 新任务标准训练 —— #
        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        # —— FDR 正则：在缓冲区样本上做分布匹配 —— #
        if not self.buffer.is_empty():
            self.opt.zero_grad()
            buf_data = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            # 稳健取法：0=examples, 2=logits（兼容多返回字段）
            buf_inputs = buf_data[0]
            buf_logits = buf_data[2]

            buf_outputs = self.net(buf_inputs)
            loss = torch.norm(self.soft(buf_outputs) - self.soft(buf_logits), p=2, dim=1).mean()
            assert not torch.isnan(loss)
            loss.backward()
            self.opt.step()

        return float(loss.item())
