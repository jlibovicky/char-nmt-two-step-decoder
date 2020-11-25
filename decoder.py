from typing import Dict, Iterable, List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import BertConfig, BertEncoder

from lee_encoder import CharToPseudoWord

T = torch.Tensor


class Decoder(nn.Module):
    def __init__(
            self,
            char_vocabulary_size: int,
            char_embedding_dim: int,
            dim: int,
            shrink_factor: int = 5,
            layers: int = 6,
            ff_dim: int = None,
            attention_heads: int = 8,
            dropout: float = 0.1) -> None:
        super(Decoder, self).__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers
        self.char_vocabulary_size = char_vocabulary_size

        self.char_embeddings = nn.Embedding(
            char_vocabulary_size, char_embedding_dim)
        self.char_encoder = CharToPseudoWord(
            char_embedding_dim, intermediate_dim=dim,
            max_pool_window=shrink_factor,
            is_decoder=True)
        config = BertConfig(
            vocab_size=dim,
            is_decoder=True,
            add_cross_attention=True,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act='relu',
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout)
        self.transformer = BertEncoder(config)

        self.char_decoder_rnn = nn.LSTM(
            char_embedding_dim, dim, batch_first=True)
        self.output_proj = nn.Linear(dim, char_vocabulary_size)


    def _hidden_states(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T) -> T:
        batch_size = target_ids.size(0)
        step_size = self.char_encoder.max_pool_window
        to_prepend = torch.ones(
            (batch_size, step_size),
            dtype=torch.int64).to(target_ids.device)
        dec_input = torch.cat([to_prepend, target_ids[:, :-step_size]], dim=1)
        input_mask = torch.cat([to_prepend, target_mask[:, :-step_size]], dim=1)

        decoder_embeddings, shrinked_mask = self.char_encoder(
            self.char_embeddings(dec_input), input_mask)

        extended_attention_mask = shrinked_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        extended_encoder_mask = (1.0 - extended_encoder_mask) * -10000.0

        decoder_states = self.transformer(
            decoder_embeddings,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=extended_encoder_mask,
            head_mask=[None] * self.layers)[0]

        return decoder_states


    def forward(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            loss_function: nn.Module) -> T:

        decoder_states = self._hidden_states(
            encoder_states, encoder_mask, target_ids, target_mask)

        losses = []
        for i, state in enumerate(decoder_states.transpose(0, 1)):
            state = state.unsqueeze(0)
            decode_start = i * self.char_encoder.max_pool_window
            state_target_ids = target_ids[:, decode_start:]
            state_target_mask = target_mask[:, decode_start:]

            if state_target_ids.size(1) == 0:
                break

            decoder_input = torch.cat((
                torch.ones((state.size(1), 1),
                            dtype=torch.int64).to(state_target_ids.device),
                state_target_ids[:, :-1]), dim=1)

            step_embedded = self.char_embeddings(decoder_input)
            step_states, _ = self.char_decoder_rnn(
                step_embedded, (state, state))
            step_logits = self.output_proj(step_states)

            loss_per_char = loss_function(
                step_logits.reshape(-1, self.char_vocabulary_size),
                state_target_ids.reshape(-1))
            losses.append(
                (loss_per_char * state_target_mask.reshape(-1)).sum() /
                state_target_mask.sum())

        return sum(losses) / len(losses)

    @torch.no_grad()
    def greedy_decode(
            self,
            encoder_states: T,
            encoder_mask: T,
            max_len: int = 500) -> Tuple[T, T]:
        batch_size = encoder_states.size(0)
        step_size = self.char_encoder.max_pool_window

        decoded = torch.ones(
            (batch_size, 0),
            dtype=torch.int64).to(encoder_states.device)

        for _ in range(max_len // step_size):
            last_state = self._hidden_states(
                encoder_states, encoder_mask,
                decoded, torch.ones_like(decoded, dtype=torch.float))[:, -1:].transpose(0, 1)

            new_chars = [
                torch.ones(
                    (batch_size, 1),
                    dtype=torch.int64).to(encoder_states.device)]
            rnn_state = (last_state, last_state)
            for _ in range(step_size):
                rnn_output, rnn_state = self.char_decoder_rnn(
                    self.char_embeddings(new_chars[-1]), rnn_state)
                next_chars = self.output_proj(rnn_output).argmax(2)
                new_chars.append(next_chars)

            to_append = torch.stack(new_chars[1:]).transpose(0, 1).squeeze(2)
            decoded = torch.cat((decoded, to_append), dim=1)

        return decoded, None
