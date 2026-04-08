import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def entropy_from_logits(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    logits: (B, C)
    返回逐样本熵 H(p) = -Σ p log p
    """
    p = F.softmax(logits, dim=1).clamp_min(eps)  # 避免 log(0)
    ent = - (p * p.log()).sum(dim=1)             # (B,)
    return ent


class ErMaxEnt(ContinualModel):
    """Experience Replay + Max-Entropy sampling."""
    NAME = 'er_maxent'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # 与原版 ER 一样默认把 buffer 放在 CPU；仅切换策略为 'maxent'
        self.buffer = Buffer(self.args.buffer_size, sample_selection_strategy='maxent')

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()

        real_batch_size = inputs.shape[0]

        # 拼接回放样本
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        # 前向 + 损失
        outputs = self.net(inputs)
        train_loss = self.loss(outputs, labels)
        train_loss.backward()
        self.opt.step()

        # 只对“当前批”的前 real_batch_size 个样本计算预测熵
        with torch.no_grad():
            cur_logits = outputs[:real_batch_size]
            per_sample_ent = entropy_from_logits(cur_logits)           # (B,)
            per_sample_ent = per_sample_ent.to(self.buffer.device)     # 与 buffer 设备一致

        # 往 buffer 写“非增强版”样本 + 标签 + 采样分数（预测熵）
        self.buffer.add_data(
            examples=not_aug_inputs,                 # 当前非增强图像
            labels=labels[:real_batch_size],         # 当前标签
            sample_selection_scores=per_sample_ent   # 预测熵作为重要性分数
        )

        return train_loss.item()
