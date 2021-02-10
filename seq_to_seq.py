"""Encoder-decoder model."""

from typing import List, Tuple

import torch
import torch.nn as nn

from lee_encoder import Encoder
from decoder import Decoder, VanillaDecoder


T = torch.Tensor

class Seq2SeqModel(nn.Module):
    def __init__(
            self, vocab_size: int,
            conv_filters: List[int],
            nar_output: bool = False,
            char_embedding_dim: int = 128,
            dim: int = 512,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            ff_dim: int = None,
            layers: int = 6,
            attention_heads: int = 8,
            dropout: float = 0.1,
            vanilla_decoder: bool = False,
            share_char_repr: bool = False) -> None:
        super().__init__()

        self.layers = layers

        self.encoder = Encoder(
            vocab_size=vocab_size,
            char_embedding_dim=char_embedding_dim,
            conv_filters=conv_filters,
            dim=dim,
            shrink_factor=shrink_factor,
            highway_layers=highway_layers,
            ff_dim=ff_dim, layers=layers,
            attention_heads=attention_heads,
            dropout=dropout,
            decoder_style_padding=share_char_repr)

        if vanilla_decoder:
            self.decoder = VanillaDecoder(
                char_vocabulary_size=vocab_size,
                dim=dim,
                layers=layers,
                ff_dim=ff_dim,
                attention_heads=attention_heads,
                dropout=dropout)
        else:
            self.decoder = Decoder(
                char_vocabulary_size=vocab_size,
                char_embedding_dim=char_embedding_dim,
                conv_filters=conv_filters,
                nar_output=nar_output,
                dim=dim,
                shrink_factor=shrink_factor,
                highway_layers=highway_layers,
                layers=layers,
                ff_dim=ff_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                encoder=self.encoder if share_char_repr else None)

    def forward(
            self, src_batch: T, src_mask: T, tgt_batch: T, tgt_mask: T,
            loss_function: nn.Module) -> T:
        encoded, enc_mask = self.encoder(src_batch, src_mask)
        loss = self.decoder(
            encoded, enc_mask, tgt_batch, tgt_mask, loss_function)

        return loss

    @torch.no_grad()
    def greedy_decode(
            self, src_batch: T, input_mask: T,
            eos_token_id: int, max_len: int = 100) -> Tuple[T, T]:
        encoder_states, encoded_mask = self.encoder(src_batch, input_mask)
        decoded, mask = self.decoder.greedy_decode(
            encoder_states, encoded_mask, eos_token_id, max_len=max_len)

        return decoded, mask
