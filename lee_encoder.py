from typing import Dict, Iterable, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_bert import BertConfig, BertEncoder

T = torch.Tensor

class Highway(nn.Module):
    """Highway layer.

    https://arxiv.org/abs/1507.06228

    Adapted from:
    https://gist.github.com/dpressel/3b4780bafcef14377085544f44183353
    """
    def __init__(self, input_size: int) -> None:
        super(Highway, self).__init__()
        self.proj = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1)
        self.transform = nn.Conv1d(
            input_size, input_size, kernel_size=1, stride=1)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, x: T) -> T:
        proj_result = F.relu(self.proj(x))
        proj_gate = torch.sigmoid(self.transform(x))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * x)
        return gated


DEFAULT_FILTERS = [200, 200, 250, 250, 300, 300, 300, 300]


class CharToPseudoWord(nn.Module):
    """Character-to-pseudoword encoder."""
    def __init__(
            self, input_dim: int, conv_filters: List[int] = DEFAULT_FILTERS,
            intermediate_dim: int = 512,
            highway_layers: int = 2, max_pool_window: int = 5,
            dropout: float = 0.1,
            is_decoder: bool = False) -> None:
        super(CharToPseudoWord, self).__init__()

        self.is_decoder = is_decoder
        self.conv_count = len(conv_filters)
        self.max_pool_window = max_pool_window
        # DO NOT PAD IN DECODER, is handled in forward
        self.convolutions = nn.ModuleList([
            nn.Conv1d(
                input_dim, dim, kernel_size=i + 1, stride=1,
                padding=(1 - int(is_decoder)) * (i + 1) // 2)
            for i, dim in enumerate(conv_filters)])

        self.cnn_output_dim = sum(conv_filters)

        self.after_cnn = nn.Sequential(
            nn.Conv1d(self.cnn_output_dim, intermediate_dim, 1, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.GroupNorm(1, intermediate_dim),
            nn.MaxPool1d(
                max_pool_window, max_pool_window,
                padding=max_pool_window // 2))

        self.highways = nn.Sequential(
            *(Highway(intermediate_dim)
              for _ in range(highway_layers)))

        self.final_mask_shrink = nn.MaxPool1d(
            max_pool_window, max_pool_window, padding=max_pool_window // 2)


    def forward(self, embedded_chars: T, mask: T) -> Tuple[T, T]:
        batch_size = embedded_chars.size(0)
        if self.is_decoder:
            padding = torch.ones(
                (batch_size, self.conv_count, embedded_chars.size(2))).to(embedded_chars.device)
            embedded_chars = torch.cat(
                [padding, embedded_chars], dim=1)
        embedded_chars = embedded_chars.transpose(2, 1)

        if self.is_decoder:
            convolved_char = torch.cat([
                conv(embedded_chars[:, :, self.conv_count - i:])
                for i, conv in enumerate(self.convolutions)], dim=1)
        else:
            convolved_char = torch.cat([
                conv(embedded_chars)[:, :, 1:] if i % 2 == 1
                else conv(embedded_chars)
                for i, conv in enumerate(self.convolutions)], dim=1)

        shrinked = self.after_cnn(convolved_char)

        output = self.highways(shrinked)
        shrinked_mask = self.final_mask_shrink(mask.unsqueeze(1)).squeeze(1)

        return output.transpose(2, 1), shrinked_mask


class Encoder(nn.Module):
    def __init__(
            self, vocab_size: int,
            char_embedding_dim: int = 128,
            dim: int = 512,
            shrink_factor: int = 5,
            ff_dim: int = None,
            layers: int = 6,
            attention_heads: int = 8,
            dropout: float = 0.1) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers
        self.layers = layers

        self.embeddings = nn.Embedding(vocab_size, char_embedding_dim)
        self.char_encoder = CharToPseudoWord(
            char_embedding_dim, intermediate_dim=dim,
            max_pool_window=shrink_factor)
        config = BertConfig(
            vocab_size=vocab_size,
            is_decoder=False,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act='relu',
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout)
        self.transformer = BertEncoder(config)

    def forward(self, data: torch.LongTensor, mask: T) -> Tuple[T, T]:
        encoded, enc_mask = self.char_encoder(self.embeddings(data), mask)

        extended_attention_mask = enc_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        transformed = self.transformer(
            encoded, attention_mask=extended_attention_mask,
            head_mask=[None] * self.layers)[0]

        return transformed, enc_mask
