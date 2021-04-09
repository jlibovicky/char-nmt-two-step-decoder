"""Simple CTC decoder for auxiliary encoder training."""

from typing import Tuple, Optional

import torch
from torch import nn


T = torch.Tensor

class CharCTCDecoder(nn.Module):
    def __init__(
            self, vocab_size: int, dim: int,
            final_stride: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.final_stride = final_stride
        self.dim = dim

        self.expand_layer = nn.Sequential(
            nn.Linear(dim, final_stride * dim))

        self.output_projection = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, vocab_size),
            nn.Dropout(dropout),
            nn.LogSoftmax(2))

        self.loss_function = nn.CTCLoss(blank=1)

    def forward(
            self,
            representation: T,
            encoder_mask: T,
            targets: torch.LongTensor,
            target_mask: T) -> Tuple[T, T, Optional[T]]:
        assert ((targets is None and target_mask is None) or
                (targets is not None and target_mask is not None))

        batch_size = representation.size(0)
        seq_lengths = encoder_mask.int().sum(1)

        states = self.expand_layer(representation).reshape(
            batch_size, -1, self.dim)

        logprobs = self.output_projection(states)
        target_lengths = target_mask.int().sum(1)
        target_lengths = torch.stack([
            target_lengths,
            torch.full_like(target_lengths, logprobs.size(1))]).min(0).values

        out_lenghts = self.final_stride * seq_lengths
        loss = self.loss_function(
            logprobs.transpose(0, 1), targets,
            input_lengths=out_lenghts,
            target_lengths=target_mask.int().sum(1))

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.zeros_like(loss)

        return logprobs, out_lenghts, loss
