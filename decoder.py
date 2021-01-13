from typing import List, Tuple

import torch
from torch import nn
from transformers.modeling_bert import BertConfig, BertModel

from lee_encoder import CharToPseudoWord, Encoder

T = torch.Tensor


class Decoder(nn.Module):
    # pylint: disable=too-many-arguments
    def __init__(
            self,
            char_vocabulary_size: int,
            char_embedding_dim: int,
            conv_filters: List[int],
            dim: int,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            layers: int = 6,
            ff_dim: int = None,
            attention_heads: int = 8,
            dropout: float = 0.1,
            max_length: int = 600,
            encoder: Encoder = None) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers
        self.char_vocabulary_size = char_vocabulary_size
        self.shrink_factor = shrink_factor

        if encoder is not None:
            self.char_embeddings = encoder.char_embeddings
            self.pre_pos_emb = encoder.pre_pos_emb
        else:
            self.char_embeddings = nn.Embedding(
                char_vocabulary_size, char_embedding_dim)
            self.pre_pos_emb = nn.Parameter(
                torch.randn(1, max_length, char_embedding_dim))
        self.char_encoder = CharToPseudoWord(
            char_embedding_dim, intermediate_dim=dim,
            conv_filters=conv_filters,
            max_pool_window=shrink_factor,
            highway_layers=highway_layers,
            is_decoder=True)

        config = BertConfig(
            vocab_size=dim,
            is_decoder=True,
            add_cross_attention=True,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act="relu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout)
        self.transformer = BertModel(config)

        self.state_to_lstm_c = nn.Sequential(
            nn.Linear(dim, char_embedding_dim),
            nn.Tanh())
        self.state_to_lstm_h = nn.Sequential(
            nn.Linear(dim, char_embedding_dim),
            nn.Tanh())

        self.char_decoder_rnn = nn.LSTM(
            char_embedding_dim, char_embedding_dim, batch_first=True)
        self.output_proj = nn.Linear(char_embedding_dim, char_vocabulary_size)
        #self.output_proj = nn.Linear(dim, char_vocabulary_size)
    # pylint: enable=too-many-arguments

    def _hidden_states(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            for_training: bool) -> T:
        """Hidden states are used as input to the decoding LSTMs."""
        batch_size = target_ids.size(0)
        to_prepend = torch.ones(
            (batch_size, self.shrink_factor),
            dtype=torch.int64).to(target_ids.device)

        if for_training:
            target_ids = target_ids[:, :-self.shrink_factor]
            target_mask = target_mask[:, :-self.shrink_factor]
        dec_input = torch.cat([to_prepend, target_ids], dim=1)
        input_mask = torch.cat([to_prepend, target_mask], dim=1)

        decoder_embeddings, shrinked_mask = self.char_encoder(
            (self.char_embeddings(dec_input) +
                self.pre_pos_emb[:, :dec_input.size(1)]),
            input_mask)

        decoder_states = self.transformer(
            input_ids=None,
            inputs_embeds=decoder_embeddings,
            attention_mask=shrinked_mask,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask)[0]

        return decoder_states

    def forward(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            loss_function: nn.Module) -> T:

        batch_size = encoder_states.size(0)
        decoder_states = self._hidden_states(
            encoder_states, encoder_mask, target_ids, target_mask,
            for_training=True)

        decoder_states_h = self.state_to_lstm_h(
            decoder_states).transpose(0, 1)
        decoder_states_c = self.state_to_lstm_c(
            decoder_states).transpose(0, 1)

        step_logits = []
        for i, (state_c, state_h) in enumerate(
                zip(decoder_states_c, decoder_states_h)):
            # unsqueeze to have time dimension first, to intialize LSTM
            # even though the RNN is batch-major
            state_c, state_h = state_c.unsqueeze(0), state_h.unsqueeze(0)

            # cut off the correct target side window
            decode_start = i * self.char_encoder.max_pool_window
            decode_end = (i + 1) * self.char_encoder.max_pool_window
            state_target_ids = target_ids[:, decode_start:decode_end]

            # prepend 1 as a start symbol for the small decoder
            small_decoder_start = torch.ones(
                (batch_size, 1), dtype=torch.int64).to(state_target_ids.device)
            decoder_input = torch.cat(
                (small_decoder_start, state_target_ids[:, :-1]), dim=1)

            step_embedded = self.char_embeddings(decoder_input)
            step_states, _ = self.char_decoder_rnn(
                step_embedded, (state_h.contiguous(), state_c.contiguous()))
            step_logits.append(self.output_proj(step_states))
            #step_logits.append(
            #    self.output_proj(decoder_states[:, i, :]).unsqueeze(1))

        decoder_logits = torch.cat(step_logits, dim=1)
        loss_per_char = loss_function(
            decoder_logits.reshape(-1, self.char_vocabulary_size),
            target_ids.reshape(-1))

        return (
            loss_per_char * target_mask.reshape(-1)).sum() / target_mask.sum()

    @torch.no_grad()
    def greedy_decode(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            max_len: int = 200) -> Tuple[T, T]:
        batch_size = encoder_states.size(0)
        step_size = self.char_encoder.max_pool_window

        decoded = torch.ones(
            (batch_size, 0),
            dtype=torch.int64).to(encoder_states.device)
        finished = torch.tensor(
            [False] * batch_size).to(encoder_states.device)

        for _ in range(max_len // step_size + 1):
            last_state = self._hidden_states(
                encoder_states, encoder_mask,
                decoded,
                torch.ones_like(decoded, dtype=torch.float),
                for_training=False)[:, -1:].transpose(0, 1)

            # this is to initiliaze the small char-level RNN
            new_chars = [
                torch.ones((batch_size, 1),
                           dtype=torch.int64).to(encoder_states.device)]
            rnn_state = (
                self.state_to_lstm_h(last_state),
                self.state_to_lstm_c(last_state))
            for _ in range(step_size):
                rnn_output, rnn_state = self.char_decoder_rnn(
                    self.char_embeddings(new_chars[-1]), rnn_state)
                next_chars = self.output_proj(rnn_output).argmax(2)
                new_chars.append(next_chars)
                finished = finished + (next_chars == eos_token_id)
            #new_chars.append(self.output_proj(last_state).argmax(2).transpose(0, 1))

                #if finished.all():
                #    break

            # we need to remove the special start for the small RNN decoder
            to_append = torch.cat(new_chars[1:], dim=1)
            decoded = torch.cat((decoded, to_append), dim=1)
            #if finished.all():
            #    break

        return decoded, None


class VanillaDecoder(nn.Module):
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
            vocab_size=dim,
            is_decoder=True,
            add_cross_attention=True,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act="relu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout)
        self.transformer = BertModel(config)

        self.output_proj = nn.Linear(dim, char_vocabulary_size)


    def _hidden_states(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            for_training: bool) -> T:
        batch_size = target_ids.size(0)
        to_prepend = torch.ones(
            (batch_size, 1),
            dtype=torch.int64).to(target_ids.device)

        if for_training:
            target_ids = target_ids[:, :-1]
            target_mask = target_mask[:, :-1]
        dec_input = torch.cat([to_prepend, target_ids], dim=1)
        input_mask = torch.cat([to_prepend, target_mask], dim=1)

        decoder_states = self.transformer(
            dec_input,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask)[0]

        return decoder_states

    def forward(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            loss_function: nn.Module) -> T:

        decoder_states = self._hidden_states(
            encoder_states, encoder_mask, target_ids, target_mask,
            for_training=True)

        decoder_logits = self.output_proj(decoder_states)

        loss_per_char = loss_function(
            decoder_logits.reshape(-1, self.char_vocabulary_size),
            target_ids.reshape(-1))

        return (
            loss_per_char * target_mask.reshape(-1)).sum() / target_mask.sum()

    @torch.no_grad()
    def greedy_decode(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            max_len: int = 200) -> Tuple[T, T]:
        batch_size = encoder_states.size(0)

        decoded = torch.ones(
            (batch_size, 0),
            dtype=torch.int64).to(encoder_states.device)
        finished = torch.tensor(
            [False] * batch_size).to(encoder_states.device)

        for _ in range(max_len):
            last_state = self._hidden_states(
                encoder_states, encoder_mask,
                decoded,
                torch.ones_like(decoded, dtype=torch.float),
                for_training=False)[:, -1]

            new_char = self.output_proj(last_state).argmax(1, keepdim=True)
            finished = finished + (new_char == eos_token_id)

            decoded = torch.cat((decoded, new_char), dim=1)
            if finished.all():
                break

        return decoded, None
