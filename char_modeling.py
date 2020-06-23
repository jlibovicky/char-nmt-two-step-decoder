"""Classes handling character-level input and outpu."""

from typing import Tuple, Optional
from operator import xor

import torch
from torch import nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    def __init__(
            self, vocab_size: int, dim: int, final_window: int,
            final_stride: int, dropout: float = 0.1) -> None:
        super().__init__()

        # TODO convolutions before doing the final one

        if final_stride < 2:
            raise ValueError(
                "The final stride needs to reduce the sequence length.")

        self.final_window = final_window
        self.final_stride = final_stride
        self.dim = dim

        self.embeddings = nn.Embedding(vocab_size, dim)
        self.final_cnn = nn.Sequential(
            nn.Conv1d(dim, dim, final_window, final_stride),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.GroupNorm(1, dim))

        self.final_mask_shrink = nn.MaxPool1d(final_window, final_stride)

    def forward(
            self, data: torch.LongTensor,
            mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO what about dropout, layer norm etc.
        x = self.embeddings(data).transpose(2, 1)
        x = self.final_cnn(x).transpose(2, 1)

        out_mask = self.final_mask_shrink(mask.unsqueeze(1)).squeeze(1)

        return x, out_mask


class CharCTCDecode(nn.Module):
    def __init__(
            self, vocab_size: int, dim: int, final_window: int,
            final_stride: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.final_window = final_window
        self.final_stride = final_stride
        self.dim = dim

        self.expand_layer = nn.Sequential(
            nn.ConvTranspose1d(dim, dim, final_window, final_stride),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, vocab_size, 1),
            nn.LogSoftmax(1))

        self.loss_function = nn.CTCLoss(blank=1)

    def forward(
            self,
            representation: torch.Tensor,
            encoder_mask: torch.Tensor,
            targets: torch.LongTensor = None,
            target_mask: torch.Tensor = None) -> Tuple[
                torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert ((targets is None and target_mask is None) or
                (targets is not None and target_mask is not None))

        seq_lengths = encoder_mask.int().sum(1)

        logprobs = self.expand_layer(
            representation.transpose(2, 1)).transpose(2, 1)

        out_lenghts = self.final_stride * seq_lengths
        loss = None
        if targets is not None and target_mask is not None:
            loss = self.loss_function(
                logprobs.transpose(0, 1), targets,
                input_lengths=out_lenghts,
                target_lengths=target_mask.int().sum(1))

        return logprobs, out_lenghts, loss
