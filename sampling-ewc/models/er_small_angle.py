# -*- coding: utf-8 -*-
"""
ER with Mid-Angle Selection
---------------------------
在标准 ER 的基础上，将“样本加入缓冲区”的策略改为 mid-angle：
- 训练时（observe）：从缓冲区取样 + 当前批次一起训练（与 ER 相同）；
- 不在 observe 中向缓冲区添加当前批次样本；
- 任务结束（end_task）：使用 fill_buffer(..., angle_mode='mid')，对“本任务训练集”
  选择与类均值余弦最接近“中位数”的样本，加入缓冲区。
"""

import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, fill_buffer


class ErSmallAngleMeanf(ContinualModel):
    """Experience Replay with MID-ANGLE selection at task end."""
    NAME = 'er_small_angle'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        # 复用 ER 的缓冲区相关参数：--buffer_size, --minibatch_size 等
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone: nn.Module, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # 训练时只读不写；任务结束时统一用 mid-angle 填充
        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        一次训练步：
        - 若缓冲区非空：取出 minibatch_size 个旧样本，与当前 inputs/labels 拼接
        - 前向/反向/更新
        - **不**向缓冲区添加当前批次（避免与 mid-angle 的“任务末统一选样”冲突）
        """
        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        return loss.item()

    def begin_task(self, dataset) -> None:
        self.net.train()

    def end_task(self, dataset) -> None:
        """
        任务结束时，用 mid-angle 从“本任务训练集”中选样加入缓冲区：
        - angle_mode='mid'：选择与类均值余弦相似度最接近“中位数”的样本；
        - normalize_features=True：先对特征做 L2 归一化，余弦计算更稳定。
        说明：fill_buffer 会根据 buffer 大小与每类目标配额为当前任务填充样本；
             之前任务加入的样本默认保留（内部会做类均衡的下采样），形成跨任务 exemplar 集。
        """
        self.net.eval()
        with torch.no_grad():
            fill_buffer(
                buffer=self.buffer,
                dataset=dataset,
                t_idx=self.current_task,
                net=self.net,
                use_herding=False,          # 关闭 herding
                angle_mode='small',           # 启用 mid-angle
                normalize_features=False,    # 先 L2 归一化再算余弦
                extend_equalize_buffer=False
            )
        self.net.train()
