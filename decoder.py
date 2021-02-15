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
            nar_output: bool = False,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            char_ff_layers: int = 2,
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
        self.nar_output = nar_output
        self.char_embedding_dim = char_embedding_dim

        if encoder is not None:
            self.char_embeddings = encoder.embeddings
            self.pre_pos_emb = encoder.pre_pos_emb
            self.char_encoder = encoder.char_encoder
        else:
            self.char_embeddings = nn.Sequential(
                nn.Embedding(char_vocabulary_size, char_embedding_dim),
                nn.Dropout(dropout))
            self.pre_pos_emb = nn.Parameter(
                torch.randn(1, max_length, char_embedding_dim))
            self.char_encoder = CharToPseudoWord(
                char_embedding_dim, intermediate_dim=dim,
                conv_filters=conv_filters,
                max_pool_window=shrink_factor,
                highway_layers=highway_layers,
                ff_layers=char_ff_layers,
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

        #self.nar_proj = nn.Sequential(
        #    nn.Linear(dim, 2 * dim),
        #    nn.ReLU(),
        #    nn.Linear(2 * dim, dim),
        #    nn.Dropout(dropout),
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, shrink_factor * char_embedding_dim))
        self.nar_proj = nn.Linear(dim, shrink_factor * char_embedding_dim)

        if not self.nar_output:
            self.char_decoder_rnn = nn.LSTM(
                2 * char_embedding_dim, 2 * char_embedding_dim, batch_first=True)
            self.output_proj = nn.Linear(3 * char_embedding_dim, char_vocabulary_size)
        else:
            self.output_proj = nn.Linear(char_embedding_dim, char_vocabulary_size)

        #self.output_proj = nn.Sequential(
        #    nn.Linear(char_embedding_dim, 2 * char_embedding_dim),
        #    nn.ReLU(),
        #    nn.Linear(2 * char_embedding_dim, char_embedding_dim),
        #    nn.Dropout(dropout),
        #    nn.LayerNorm(char_embedding_dim),
        #    nn.Linear(char_embedding_dim, char_vocabulary_size))
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
            #(batch_size, self.shrink_factor),
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

        # Char states are now a multiply of the `shrink_factor`
        # which might include at most `shrink_factor - 1` paddings
        # for which there are not labels
        char_states = self.nar_proj(decoder_states).reshape(
            batch_size, -1, self.char_embedding_dim)[:, :target_ids.size(1)]

        if self.nar_output:
            decoder_logits = self.output_proj(char_states)
            loss_per_char = loss_function(
                decoder_logits.reshape(-1, self.char_vocabulary_size),
                target_ids.reshape(-1))

            return (
                (loss_per_char * target_mask.reshape(-1)).sum() /
                target_mask.sum())

        # DECODING WITH LSTM
        small_decoder_start = torch.ones(
            (batch_size, 1), dtype=torch.int64).to(target_ids.device)
        pad_target_ids = torch.cat(
            (small_decoder_start, target_ids), dim=1)

        loss_sum = 0
        mask_sum = 0
        rnn_state = None
        for i in range(decoder_states.size(1)):
            # cut off the correct target side window
            decode_start = i * self.shrink_factor
            decode_end = min(
                target_mask.size(1),
                (i + 1) * self.char_encoder.max_pool_window)
            step_input_ids = pad_target_ids[:, decode_start:decode_end]
            step_output_ids = pad_target_ids[:, decode_start + 1: decode_end + 1]

            step_mask = target_mask[:, decode_start:decode_end]
            step_char_states = char_states[:, decode_start:decode_end]

            step_embedded = self.char_embeddings(step_input_ids)
            step_states, rnn_state = self.char_decoder_rnn(
                torch.cat((step_embedded, step_char_states), dim=2),
                rnn_state)
            this_step_logits = self.output_proj(
                torch.cat([step_states, step_char_states], dim=2))

            step_loss = loss_function(
                this_step_logits.reshape(-1, self.char_vocabulary_size),
                step_output_ids.reshape(-1))
            loss_sum += (step_loss * step_mask.reshape(-1)).sum()
            mask_sum += step_mask.sum()
        return loss_sum / mask_sum

    @torch.no_grad()
    def greedy_decode(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            max_len: int = 300) -> Tuple[T, T]:
        batch_size = encoder_states.size(0)
        step_size = self.char_encoder.max_pool_window

        decoded = torch.ones(
            (batch_size, 0),
            dtype=torch.int64).to(encoder_states.device)
        next_chars = torch.ones(
            (batch_size, 1),
            dtype=torch.int64).to(encoder_states.device)
        # pylint: disable=not-callable
        finished = torch.tensor(
            [False] * batch_size).to(encoder_states.device)
        # pylint: enable=not-callable

        rnn_state = None
        for _ in range(max_len // step_size + 1):
            last_state = self._hidden_states(
                encoder_states, encoder_mask,
                decoded,
                torch.ones_like(decoded, dtype=torch.float),
                for_training=False)[:, -1:]
            char_states = self.nar_proj(last_state).reshape(
                batch_size, -1, self.char_embedding_dim)

            if self.nar_output:
                decoder_logits = self.output_proj(char_states)
                chars_to_append = decoder_logits.argmax(2)
            else:
                last_state = last_state.transpose(0, 1)
                # this is to initiliaze the small char-level RNN
                new_chars = [decoded[:, -1:]]
                for i in range(step_size):
                    embeded_prev = self.char_embeddings(next_chars)
                    rnn_input = torch.cat(
                        (embeded_prev, char_states[:, i:i+1]), dim=2)
                    rnn_output, rnn_state = self.char_decoder_rnn(
                        rnn_input, rnn_state)
                    next_chars = self.output_proj(
                        torch.cat([rnn_output, char_states[:, i:i+1]], dim=2)).argmax(2)
                    new_chars.append(next_chars)
                    finished = finished + (next_chars == eos_token_id)

                    if finished.all():
                        break
                # we need to remove the special start for the small RNN decoder
                chars_to_append = torch.cat(new_chars[1:], dim=1)
            decoded = torch.cat((decoded, chars_to_append), dim=1)
            if finished.all():
                break

        return decoded[:, 1:], None


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
        # pylint: disable=not-callable
        finished = torch.tensor(
            [False] * batch_size).to(encoder_states.device)
        # pylint: enable=not-callable

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
