"""Encoder-decoder model."""

from typing import List, Tuple

import torch
import torch.nn as nn

from encoder import Encoder, VanillaEncoder
from decoder import Decoder, VanillaDecoder


T = torch.Tensor


def compute_attention_entropy(
        att_matrix: T, query_mask: T, key_mask: T) -> float:
    # att matrix is: batch x heads x q_len x k_len

    # first entropy of each distribution, non-existing key positions
    # must be asked out
    prenorm_entropies = -(torch.log(att_matrix) * att_matrix)
    prenorm_entropies[prenorm_entropies.isnan()] = 0.0
    distr_entropies = prenorm_entropies.sum(3)
    # shape: batch x head x q_len

    # now average over relevant query positions
    batch_head_entropies = (
        distr_entropies * query_mask.unsqueeze(1)).sum(2) / query_mask.sum()

    return batch_head_entropies.mean(0).mean(0)


class Seq2SeqModel(nn.Module):
    def __init__(
            self, vocab_size: int,
            conv_filters: List[int],
            nar_output: bool = False,
            char_embedding_dim: int = 128,
            dim: int = 512,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            char_ff_layers: int = 2,
            ff_dim: int = None,
            layers: int = 6,
            attention_heads: int = 8,
            dropout: float = 0.1,
            vanilla_encoder: bool = False,
            vanilla_decoder: bool = False,
            share_char_repr: bool = False) -> None:
        super().__init__()

        self.layers = layers

        if vanilla_encoder:
            self.encoder = VanillaEncoder(
                char_vocabulary_size=vocab_size,
                dim=dim,
                layers=layers,
                ff_dim=ff_dim,
                attention_heads=attention_heads,
                dropout=dropout)
        else:
            self.encoder = Encoder(
                vocab_size=vocab_size,
                char_embedding_dim=char_embedding_dim,
                conv_filters=conv_filters,
                dim=dim,
                shrink_factor=shrink_factor,
                highway_layers=highway_layers,
                char_ff_layers=char_ff_layers,
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
                dropout=dropout,
                encoder=self.encoder if (
                    share_char_repr and vanilla_encoder) else None)
        else:
            self.decoder = Decoder(
                char_vocabulary_size=vocab_size,
                char_embedding_dim=char_embedding_dim,
                conv_filters=conv_filters,
                nar_output=nar_output,
                dim=dim,
                shrink_factor=shrink_factor,
                highway_layers=highway_layers,
                char_ff_layers=char_ff_layers,
                layers=layers,
                ff_dim=ff_dim,
                attention_heads=attention_heads,
                dropout=dropout,
                encoder=self.encoder if share_char_repr else None)

    def forward(
            self, src_batch: T, src_mask: T, tgt_batch: T, tgt_mask: T,
            loss_function: nn.Module, log_details: bool = False) -> T:
        encoded, enc_mask, enc_attention = self.encoder(src_batch, src_mask)
        loss, details = self.decoder(
            encoded, enc_mask, tgt_batch, tgt_mask, loss_function,
            log_details=log_details)

        if log_details:
            details["enc_attentions"] = enc_attention
            details["enc_attention_entropies"] = [
                compute_attention_entropy(att, enc_mask, enc_mask)
                for att in enc_attention]
            shrinked_mask = details["decoder_mask"]
            details["dec_attention_entropies"] = [
                compute_attention_entropy(att, shrinked_mask, shrinked_mask)
                for att in details["decoder_self_attention"]]
            details["encdec_attention_entropies"] = [
                compute_attention_entropy(att, shrinked_mask, enc_mask)
                for att in details["decoder_self_attention"]]

        return loss, details

    @torch.no_grad()
    def greedy_decode(
            self, src_batch: T, input_mask: T,
            eos_token_id: int, max_len: int = 400) -> Tuple[T, T]:
        encoder_states, encoded_mask, _ = self.encoder(src_batch, input_mask)
        decoded, mask = self.decoder.greedy_decode(
            encoder_states, encoded_mask, eos_token_id, max_len=max_len)

        return decoded, mask

    @torch.no_grad()
    def beam_search(
            self, src_batch: T, input_mask: T,
            eos_token_id: int,
            beam_size: int = 5,
            len_norm: float = 0.5,
            max_len: int = 400) -> Tuple[T, T]:
        encoder_states, encoded_mask, _ = self.encoder(src_batch, input_mask)
        decoded, mask = self.decoder.beam_search(
            encoder_states, encoded_mask, eos_token_id,
            beam_size=beam_size, len_norm=len_norm, max_len=max_len)

        return decoded, mask

    @property
    def char_level_param_count(self) -> int:
        """Number of parameters in character processing layers."""

        relevant_parts = [self.encoder.embeddings]

        if isinstance(self.encoder, Encoder):
            relevant_parts.append(self.encoder.char_encoder)

        if isinstance(self.decoder, VanillaDecoder):
            relevant_parts.append(self.decoder.transformer.embeddings)
        else:
            relevant_parts.extend([
                self.decoder.nar_proj, self.decoder.output_proj])
            if not self.decoder.nar_output:
                relevant_parts.append(self.decoder.char_decoder_rnn)
            if not self.decoder.char_embeddings not in relevant_parts:
                relevant_parts.extend([
                    self.decoder.char_embeddings, self.decoder.char_encoder])

        return sum(p.numel() for part in relevant_parts
                   for p in part.parameters())
