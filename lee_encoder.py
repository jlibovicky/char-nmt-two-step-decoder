from typing import Dict, Iterable, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import BertConfig, BertEncoder

T = torch.Tensor

class Highway(nn.Module):
    """Highway layer.

    https://arxiv.org/abs/1507.06228

    Taken from:
    https://gist.github.com/dpressel/3b4780bafcef14377085544f44183353
    """
    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.proj = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1)
        self.transform = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, input):
        proj_result = F.relu(self.proj(input))
        proj_gate = torch.sigmoid(self.transform(input))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * input)
        return gated


DEFAULT_FILTERS = [200, 200, 250, 250, 300, 300, 300, 300]


class CharToPseudoWord(nn.Module):
    """Character-to-pseudoword encoder.
    """
    def __init__(
            self, input_dim: int, conv_filters: List[int] = DEFAULT_FILTERS,
            intermediate_dim: int=512,
            highway_layers: int=2, max_pool_window: int=5, dropout=0.1) -> None:
        super(CharToPseudoWord, self).__init__()

        self.convolutions = nn.ModuleList([
            nn.Conv1d(input_dim, dim, kernel_size=i + 1, stride=1, padding=(i + 1) // 2)
            for i, dim in enumerate(conv_filters)])

        self.cnn_output_dim = sum(conv_filters)

        self.after_cnn = nn.Sequential(
            nn.Conv1d(self.cnn_output_dim, intermediate_dim, 1, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.GroupNorm(1, intermediate_dim),
            nn.MaxPool1d(max_pool_window, max_pool_window))

        self.highways = nn.Sequential(
            *(Highway(intermediate_dim)
                for _ in range(highway_layers)))

        self.final_mask_shrink = nn.MaxPool1d(max_pool_window, max_pool_window)


    def forward(self, embedded_chars: T, mask: T) -> Tuple[T, T]:
        embedded_chars = embedded_chars.transpose(2, 1)

        convolved_char = torch.cat([
            conv(embedded_chars)[:, :, 1:] if i % 2 == 1
            else conv(embedded_chars)
            for i, conv in enumerate(self.convolutions)], dim=1)

        shrinked = self.after_cnn(convolved_char)

        output = self.highways(shrinked)
        shrinked_mask = self.final_mask_shrink(mask.unsqueeze(1)).squeeze(1)

        return output.transpose(2, 1), shrinked_mask

