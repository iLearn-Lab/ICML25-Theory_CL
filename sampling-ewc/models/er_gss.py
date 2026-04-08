# models/er_gss.py
# Copyright ...
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser

# 关键：使用 GSS 的 Buffer 实现
from utils.gss_buffer import Buffer as GSSBuffer


class ErGSS(ContinualModel):
    """Experience Replay with GSS storage strategy (gradient-diversity based selection)."""
    NAME = 'er_gss'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        # GSS 需要一个用于梯度对比的 mini-batch 大小；缺省用普通 minibatch_size
        parser.add_argument('--gss_minibatch_size', type=int, default=None,
                            help='Batch size used for gradient comparison in GSS buffer.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        # 和 GSS 一样初始化 Buffer（传入模型自身以便调用 get_grads）
        gss_mb = self.args.gss_minibatch_size if self.args.gss_minibatch_size is not None else self.args.minibatch_size
        self.buffer = GSSBuffer(self.args.buffer_size, self.device, gss_mb, self)

    # —— 关键：提供给 GSSBuffer 使用的梯度提取函数，等同于 GSS 模型里的实现 ——
    def get_grads(self, inputs, labels):
        """
        Return flattened per-batch gradient vector(s) for (inputs, labels).
        GSSBuffer 会调用它来计算样本梯度并做多样性选择。
        """
        self.net.eval()
        self.opt.zero_grad()

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()

        grads = self.net.get_grads().clone().detach()
        self.opt.zero_grad()
        self.net.train()

        if grads.ndim == 1:
            grads = grads.unsqueeze(0)
        return grads

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        训练保持与 ER 一致：当前 batch 与 buffer batch 拼接训练；
        区别仅在于：训练后把“未增强的新样本”交给 GSSBuffer.add_data()，由其做梯度多样性选择。
        """
        real_batch_size = inputs.size(0)

        # 可选：与 GSS 对齐，清理内部缓存/临时状态
        if hasattr(self.buffer, "drop_cache"):
            self.buffer.drop_cache()
        if hasattr(self.buffer, "reset_fathom"):
            self.buffer.reset_fathom()

        self.opt.zero_grad()

        if not self.buffer.is_empty():
            # 与 ER 一样：从 buffer 取一批拼接训练
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform
            )
            inputs = torch.cat([inputs, buf_inputs])
            labels = torch.cat([labels, buf_labels])

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        # 只把“真实新样本”（未增强版本）尝试写入 Buffer（GSS 内部决定去留/替换位）
        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size]
        )
        return loss.item()
