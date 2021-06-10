from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_bert import BertConfig, BertModel # type: ignore

T = torch.Tensor

class Highway(nn.Module):
    """Highway layer.

    https://arxiv.org/abs/1507.06228

    Adapted from:
    https://gist.github.com/dpressel/3b4780bafcef14377085544f44183353
    """
    def __init__(self, input_size: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv1d(input_size, input_size, kernel_size=1, stride=1)
        self.transform = nn.Conv1d(
            input_size, input_size, kernel_size=1, stride=1)
        self.transform.bias.data.fill_(-2.0) # type: ignore
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: T) -> T:
        proj_result = F.relu(self.proj(x))
        proj_gate = torch.sigmoid(self.transform(x))
        gated = (proj_gate * proj_result) + ((1 - proj_gate) * x)
        return gated


class TransformerFeedForward(nn.Module):
    """Feedforward sublayer from the Transformer."""

    def __init__(
            self, input_size: int,
            intermediate_size: int, dropout: float) -> None:
        super().__init__()

        self.sublayer = nn.Sequential(
            nn.Linear(input_size, intermediate_size),
            nn.ReLU(),
            nn.Linear(intermediate_size, input_size),
            nn.Dropout(dropout))

        self.norm = nn.LayerNorm(input_size)

    def forward(self, input_tensor: T) -> T:
        output = self.sublayer(input_tensor)
        return self.norm(output + input_tensor)


DEFAULT_FILTERS = [128, 256, 512, 512]


class CharToPseudoWord(nn.Module):
    """Character-to-pseudoword encoder."""
    # pylint: disable=too-many-arguments
    def __init__(
            self, input_dim: int,
            # pylint: disable=dangerous-default-value
            conv_filters: List[int] = DEFAULT_FILTERS,
            # pylint: enable=dangerous-default-value
            intermediate_cnn_layers: int = 0,
            intermediate_dim: int = 512,
            highway_layers: int = 2,
            ff_layers: int = 2,
            max_pool_window: int = 5,
            dropout: float = 0.1,
            is_decoder: bool = False) -> None:
        super().__init__()

        self.is_decoder = is_decoder
        self.conv_count = len(conv_filters)
        self.max_pool_window = max_pool_window
        # DO NOT PAD IN DECODER, is handled in forward
        if conv_filters == [0]:
            self.convolutions = None
            self.conv_count = 0
            self.cnn_output_dim = input_dim
        else:
            # TODO maybe the correct padding for the decoder is just 2 * i
            self.convolutions = nn.ModuleList([
                nn.Conv1d(
                    input_dim, dim, kernel_size=2 * i + 1, stride=1,
                    padding=2 * i if is_decoder else i)
                for i, dim in enumerate(conv_filters)])
            self.cnn_output_dim = sum(conv_filters)

        self.after_cnn = nn.Sequential(
            nn.Conv1d(self.cnn_output_dim, intermediate_dim, 1),
            nn.Dropout(dropout),
            nn.ReLU())

        self.intermediate_cnns = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(intermediate_dim, 2 * intermediate_dim, 3, padding=1),
                nn.Dropout(dropout))
            for _ in range(intermediate_cnn_layers)])
        self.intermediate_cnn_norm = nn.ModuleList([
            nn.LayerNorm(intermediate_dim)
            for _ in range(intermediate_cnn_layers)])

        self.shrink = nn.MaxPool1d(
            max_pool_window, max_pool_window,
            padding=0, ceil_mode=True)

        self.highways = nn.Sequential(
            *(Highway(intermediate_dim, dropout)
              for _ in range(highway_layers)))

        self.after_highways = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(intermediate_dim))

        self.ff_layers = nn.Sequential(
            *(TransformerFeedForward(
                intermediate_dim, 2 * intermediate_dim, dropout)
              for _ in range(ff_layers)))

        self.final_mask_shrink = nn.MaxPool1d(
            max_pool_window, max_pool_window, padding=0, ceil_mode=True)
    # pylint: enable=too-many-arguments


    def forward(self, embedded_chars: T, mask: T) -> Tuple[T, T]:
        embedded_chars = embedded_chars.transpose(2, 1)
        conv_mask = mask.unsqueeze(1)

        if self.convolutions is not None:
            conv_outs = []
            for i, conv in enumerate(self.convolutions):
                conv_i_out = conv(embedded_chars * conv_mask)
                if self.is_decoder and i > 0:
                    conv_i_out = conv_i_out[:, :, :-2 * i]
                conv_outs.append(conv_i_out)

            convolved_char = torch.cat(conv_outs, dim=1)
        else:
            convolved_char = embedded_chars

        convolved_char = self.after_cnn(convolved_char)
        for cnn, norm in zip(self.intermediate_cnns,
                             self.intermediate_cnn_norm):
            conv_out = F.glu(cnn(convolved_char * conv_mask), dim=1)
            convolved_char = norm(
                (conv_out + convolved_char).transpose(2, 1)).transpose(2, 1)

        shrinked = self.shrink(convolved_char)
        output = self.highways(shrinked).transpose(2, 1)
        output = self.after_highways(output)
        output = self.ff_layers(output)
        shrinked_mask = self.final_mask_shrink(mask.unsqueeze(1)).squeeze(1)

        return output, shrinked_mask


class Encoder(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(
            self, vocab_size: int,
            char_embedding_dim: int = 128,
            # pylint: disable=dangerous-default-value
            conv_filters: List[int] = DEFAULT_FILTERS,
            # pylint: enable=dangerous-default-value
            intermediate_cnn_layers: int = 0,
            dim: int = 512,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            char_ff_layers: int = 2,
            ff_dim: int = None,
            layers: int = 6,
            attention_heads: int = 8,
            dropout: float = 0.1,
            max_length: int = 600,
            decoder_style_padding: bool = False) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers

        self.embeddings = nn.Embedding(vocab_size, char_embedding_dim)
        self.pre_pos_emb = nn.Parameter(
            torch.randn(1, max_length, char_embedding_dim))
        self.char_encoder = CharToPseudoWord(
            char_embedding_dim, intermediate_dim=dim,
            conv_filters=conv_filters,
            intermediate_cnn_layers=intermediate_cnn_layers,
            highway_layers=highway_layers,
            ff_layers=char_ff_layers,
            max_pool_window=shrink_factor,
            dropout=dropout,
            is_decoder=decoder_style_padding)
        config = BertConfig(
            vocab_size=vocab_size,
            is_decoder=False,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act="relu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            output_attentions=True)
        self.transformer = BertModel(config)
    # pylint: enable=too-many-arguments

    def forward(self, data: torch.LongTensor, mask: T) -> Tuple[T, T, T]:
        encoded, enc_mask = self.char_encoder(
            self.embeddings(data),# + self.pre_pos_emb[:, :data.size(1)],
            mask)

        transformed, _, attentions = self.transformer(
            input_ids=None,
            inputs_embeds=encoded,
            attention_mask=enc_mask)

        return transformed, enc_mask, attentions


class VanillaEncoder(nn.Module):
    def __init__(
            self,
            char_vocabulary_size: int,
            dim: int,
            layers: int = 6,
            ff_dim: int = None,
            attention_heads: int = 8,
            dropout: float = 0.1) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers
        self.char_vocabulary_size = char_vocabulary_size

        config = BertConfig(
            vocab_size=char_vocabulary_size,
            is_decoder=False,
            add_cross_attention=False,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act="relu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            output_attentions=True)
        self.transformer = BertModel(config)

    @property
    def embeddings(self) -> nn.Module:
        return self.transformer.embeddings.word_embeddings

    def forward(self, data: torch.LongTensor, mask: T) -> Tuple[T, T, T]:
        transformed, _, attentions = self.transformer(
            input_ids=data,
            attention_mask=mask)
        return transformed, mask, attentions
