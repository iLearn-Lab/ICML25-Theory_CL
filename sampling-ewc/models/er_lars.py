"""
Experience Replay with LARS sampling.
- Replaces the buffer's sampling strategy with LARS.
- Uses per-sample training loss as importance scores when adding new data to the buffer.

Usage:
    --model er_lars
    (other ER args remain the same, e.g., --buffer_size, --minibatch_size)
"""

# Copyright 2020-present,
#   Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# Modifications 2025-present for LARS integration on ER by <your name>.
# Licensed under the license in the root LICENSE file.

import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class ErLARS(ContinualModel):
    """Continual learning via Experience Replay + LARS sampling."""
    NAME = 'er_lars'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        """
        Returns an ArgumentParser with rehearsal-related arguments.
        """
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        """
        ER with LARS:
        - Buffer uses sample_selection_strategy='lars'
        - During add_data, we pass per-sample importance scores to update LARS state.
        """
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        # Keep buffer on CPU to save VRAM (like the original ER); just switch strategy to 'lars'.
        # If you prefer keeping importance scores on GPU, pass device=self.device here.
        self.buffer = Buffer(self.args.buffer_size, sample_selection_strategy='lars')

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        """
        One training step:
        1) Optionally sample a minibatch from buffer (with transform) and concatenate.
        2) Forward, compute training loss on the concatenated batch, backward, step.
        3) Add the current (non-augmented) batch to buffer with per-sample loss as LARS scores.
        """
        self.opt.zero_grad()

        # Original current-batch size (before concatenation with replayed samples)
        real_batch_size = inputs.shape[0]

        # If buffer is not empty, fetch a replay minibatch (already moved to self.device inside get_data)
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device
            )
            inputs = torch.cat((inputs, buf_inputs), dim=0)
            labels = torch.cat((labels, buf_labels), dim=0)

        # Forward on the concatenated batch
        outputs = self.net(inputs)

        # Training loss for the whole (current + replay) batch
        train_loss = self.loss(outputs, labels)
        train_loss.backward()
        self.opt.step()

        # ----- LARS importance scores for sample selection -----
        # Use per-sample cross-entropy on CURRENT samples only (the first real_batch_size items)
        with torch.no_grad():
            # Robustly compute per-sample scores.
            # Assumes standard classification (logits) with integer labels.
            # If your loss is not CE, importance as CE still works as a proxy for "hardness".
            per_sample_scores = F.cross_entropy(
                outputs[:real_batch_size],
                labels[:real_batch_size],
                reduction='none'
            ).detach()

            # Move scores to buffer's device (buffer is on CPU by default)
            per_sample_scores = per_sample_scores.to(self.buffer.device)

        # Store the *non-augmented* current inputs with labels & LARS scores
        # Only the first real_batch_size correspond to the current task batch.
        self.buffer.add_data(
            examples=not_aug_inputs,                    # unaugmented current inputs
            labels=labels[:real_batch_size],            # current labels only
            sample_selection_scores=per_sample_scores   # importance for LARS
        )

        return train_loss.item()
