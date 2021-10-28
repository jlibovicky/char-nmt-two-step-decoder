from typing import Any, Dict, List, Optional, Tuple

from charformer_pytorch import GBST
import torch
from torch import nn
from torch.functional import F
from transformers.modeling_bert import BertConfig, BertModel

from canine import CanineEncoder
from encoder import CharToPseudoWord, Encoder, VanillaEncoder

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
            charformer_block_size: int = 5,
            highway_layers: int = 2,
            char_ff_layers: int = 2,
            layers: int = 6,
            ff_dim: int = None,
            attention_heads: int = 8,
            dropout: float = 0.1,
            max_length: int = 600,
            encoder: Encoder = None,
            char_process_type: str = "conv") -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers
        self.char_vocabulary_size = char_vocabulary_size
        self.shrink_factor = shrink_factor
        self.nar_output = nar_output
        self.char_embedding_dim = char_embedding_dim
        self.char_process_type = char_process_type

        if encoder is not None:
            self.char_embeddings: nn.Module = encoder.embeddings
            self.pre_pos_emb = encoder.pre_pos_emb
            self.char_encoder = encoder.char_encoder
        else:
            if char_process_type == "conv":
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
            elif char_process_type == "charformer":
                self.char_encoder = GBST(
                    num_tokens=char_vocabulary_size,
                    dim=dim,
                    max_block_size=charformer_block_size,
                    downsample_factor=shrink_factor,
                    score_consensus_attn=True)
            elif char_process_type == "canine":
                self.embeddings = nn.Embedding(char_vocabulary_size, dim)
                self.char_encoder = CanineEncoder(
                    hidden_size=dim,
                    num_attention_heads=attention_heads,
                    dropout=dropout,
                    shrink_factor=shrink_factor,
                    attend_from_chunk_width=shrink_factor,
                    attend_to_chunk_width=shrink_factor,
                    attend_from_chunk_stride=shrink_factor,
                    attend_to_chunk_stride=shrink_factor)
            else:
                raise ValueError()

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
            attention_probs_dropout_prob=dropout,
            output_attentions=True)
        self.transformer = BertModel(config)

        #self.nar_proj = nn.Sequential(
        #    nn.Linear(dim, 2 * dim),
        #    nn.ReLU(),
        #    nn.Linear(2 * dim, dim),
        #    nn.Dropout(dropout),
        #    nn.LayerNorm(dim),
        #    nn.Linear(dim, shrink_factor * char_embedding_dim))
        self.nar_proj = nn.Linear(dim, shrink_factor * dim // 2)

        if not self.nar_output:
            self.char_decoder_rnn = nn.LSTM(
                char_embedding_dim +  dim // 2,
                dim, batch_first=True)
            self.output_proj = nn.Linear(
                dim // 2 + dim, char_vocabulary_size)
        else:
            self.output_proj = nn.Linear(
                dim // 2, char_vocabulary_size)

        #self.output_proj = nn.Sequential(
        #    nn.Linear(char_embedding_dim, 2 * char_embedding_dim),
        #    nn.ReLU(),
        #    nn.Linear(2 * char_embedding_dim, char_embedding_dim),
        #    nn.Dropout(dropout),
        #    nn.LayerNorm(char_embedding_dim),
        #    nn.Linear(char_embedding_dim, char_vocabulary_size))
    # pylint: enable=too-many-arguments

    @property
    def embeddings(self) -> nn.Module:
        if isinstance(self.char_embedding, nn.Sequential):
            return self.char_embeddings[0] # type: ignore
        return self.char_embeddings

    def _hidden_states(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            for_training: bool) -> Tuple[T, T, T, T]:
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

        if self.char_process_type in ["conv", "canine"]:
            decoder_embeddings, shrinked_mask = self.char_encoder(
                (self.char_embeddings(dec_input) +
                    self.pre_pos_emb[:, :dec_input.size(1)]),
                input_mask)
        elif self.char_process_type == "charformer":
            decoder_embeddings, shrinked_mask_bool = self.char_encoder(
                dec_input, input_mask.bool())
            shrinked_mask = shrinked_mask_bool.float()
        else:
            raise ValueError(f"Invalid char_process_type '{self.char_process_type}'")

        decoder_states, _, self_att, encdec_att = self.transformer(
            input_ids=None,
            inputs_embeds=decoder_embeddings,
            attention_mask=shrinked_mask,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask)

        return decoder_states, shrinked_mask, self_att, encdec_att

    def forward(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            loss_function: nn.Module,
            log_details: bool = False) -> Tuple[T, Optional[Dict[str, Any]]]:

        details: Optional[Dict[str, Any]] = None
        if log_details:
            details = {}

        batch_size = encoder_states.size(0)
        decoder_states, shrinked_mask, self_att, encdec_att = self._hidden_states(
            encoder_states, encoder_mask, target_ids, target_mask,
            for_training=True)

        # Char states are now a multiply of the `shrink_factor`
        # which might include at most `shrink_factor - 1` paddings
        # for which there are not labels
        char_states = self.nar_proj(decoder_states).reshape(
            batch_size, -1, self.dim // 2)[:, :target_ids.size(1)]

        if self.nar_output:
            decoder_logits = self.output_proj(char_states)
            loss_per_char = loss_function(
                decoder_logits.reshape(-1, self.char_vocabulary_size),
                target_ids.reshape(-1))

            return (
                (loss_per_char * target_mask.reshape(-1)).sum() /
                target_mask.sum(), details)

        # DECODING WITH LSTM
        small_decoder_start = torch.ones(
            (batch_size, 1), dtype=torch.int64).to(target_ids.device)
        pad_target_ids = torch.cat(
            (small_decoder_start, target_ids), dim=1)

        loss_sum = 0
        mask_sum = 0
        rnn_state = None
        entropy_sum = 0. # type: ignore
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

            if log_details:
                step_entropies = -(
                    F.log_softmax(this_step_logits, dim=1) *
                    F.softmax(this_step_logits, dim=-1)).sum(2)
                entropy_sum += (step_entropies * step_mask).sum()

        if details is not None:
            details["output_entropy"] = entropy_sum / mask_sum
            details["decoder_self_attention"] = self_att
            details["ecnoder_decoder_attention"] = encdec_att
            details["decoder_mask"] = shrinked_mask

        return loss_sum / mask_sum, details

    @torch.no_grad()
    def greedy_decode(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            max_len: int = 300,
            sample: bool = False) -> Tuple[T, T]:
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
            states, _, _, _ = self._hidden_states(
                encoder_states, encoder_mask,
                decoded,
                torch.ones_like(decoded, dtype=torch.float),
                for_training=False)
            last_state = states[:, -1:]
            char_states = self.nar_proj(last_state).reshape(
                batch_size, -1, self.dim // 2)

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
                    out_logits = self.output_proj(
                        torch.cat(
                            [rnn_output, char_states[:, i:i+1]], dim=2))
                    if sample:
                        out_dist = F.softmax(out_logits, dim=2)
                        next_chars = torch.multinomial(out_dist.squeeze(1), 1)
                    else:
                        next_chars = out_logits.argmax(2)
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

    @torch.no_grad()
    def beam_search(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            beam_size: int,
            len_norm: float,
            sample: bool = False,
            max_len: int = 300) -> Tuple[T, T]:
        device = encoder_states.device
        batch_size = encoder_states.size(0)

        cur_len = 0
        current_beam = 1

        decoded = torch.ones(
            (batch_size, 1, 1), dtype=torch.long).to(device)
        finished = torch.full(
            (batch_size, 1, 0), False, dtype=torch.bool).to(device)
        scores = torch.zeros((batch_size, 1)).to(device)
        rnn_state = None

        flat_decoded = decoded.squeeze(1)
        flat_finished = finished.squeeze(1)
        while cur_len < max_len:
            states, _, _, _ = self._hidden_states(
                encoder_states, encoder_mask,
                flat_decoded[:, 1:],
                1 - flat_finished.float(),
                for_training=False)
            last_state = states[:, -1:]
            char_states = self.nar_proj(last_state).reshape(
                batch_size * current_beam,
                self.shrink_factor, self.char_embedding_dim)

            for i in range(self.shrink_factor):
                embeded_prev = self.char_embeddings(flat_decoded[:, -1:])
                rnn_input = torch.cat(
                    (embeded_prev, char_states[:, i:i+1]), dim=2)
                rnn_output, rnn_state = self.char_decoder_rnn(
                    rnn_input, rnn_state)
                next_char_logprobs = F.log_softmax(self.output_proj(
                    torch.cat([rnn_output, char_states[:, i:i+1]], dim=2)),
                    dim=2)

                # get scores of all expanded hypotheses
                candidate_scores = (
                    scores.unsqueeze(2) +
                    next_char_logprobs.reshape(batch_size, current_beam, -1))
                norm_factor = torch.pow(
                    (1 - finished.float()).sum(2, keepdim=True) + 1, len_norm)
                normed_scores = candidate_scores / norm_factor

                # reshape for beam members and get top k
                _, best_indices = normed_scores.reshape(
                    batch_size, -1).topk(beam_size, dim=-1)
                next_char_ids = best_indices % self.char_vocabulary_size
                hypothesis_ids = best_indices // self.char_vocabulary_size

                # numbering elements in the extended batch, i.e. beam size
                # copies of each batch element
                beam_offset = torch.arange(
                    0, batch_size * current_beam, step=current_beam,
                    dtype=torch.long, device=device)
                global_best_indices = (
                    beam_offset.unsqueeze(1) + hypothesis_ids).reshape(-1)

                # now select appropriate histories
                decoded = torch.cat((
                    flat_decoded.index_select(
                        0, global_best_indices).reshape(
                            batch_size, beam_size, -1),
                    next_char_ids.unsqueeze(-1)), dim=2)

                reordered_finished = flat_finished.index_select(
                    0, global_best_indices).reshape(batch_size, beam_size, -1)
                finished_now = (next_char_ids == eos_token_id)
                if reordered_finished.size(2) > 0:
                    finished_now += reordered_finished[:, :, -1]
                finished = torch.cat((
                    reordered_finished,
                    finished_now.unsqueeze(-1)), dim=2)

                if finished_now.all():
                    break

                char_states = char_states.index_select(0, global_best_indices)
                rnn_state = (
                    rnn_state[0].index_select(1, global_best_indices),
                    rnn_state[1].index_select(1, global_best_indices))

                # re-order scores
                scores = candidate_scores.reshape(
                    batch_size, -1).gather(-1, best_indices)

                # tile encoder after first step
                if cur_len == 1:
                    encoder_states = encoder_states.unsqueeze(1).repeat(
                        1, beam_size, 1, 1).reshape(
                            batch_size * beam_size, encoder_states.size(1),
                            encoder_states.size(2))
                    encoder_mask = encoder_mask.unsqueeze(1).repeat(
                        1, beam_size, 1).reshape(batch_size * beam_size, -1)

                # in the first iteration, beam size is 1, in the later ones,
                # it is the real beam size
                current_beam = beam_size
                cur_len += 1

                flat_decoded = decoded.reshape(-1, cur_len + 1)
                flat_finished = finished.reshape(-1, cur_len)

            if finished_now.all():
                break

        return (decoded[:, 0], finished[:, 0].logical_not())


class VanillaDecoder(nn.Module):
    def __init__(
            self,
            char_vocabulary_size: int,
            dim: int,
            layers: int = 6,
            ff_dim: int = None,
            attention_heads: int = 8,
            dropout: float = 0.1,
            encoder: VanillaEncoder = None) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers
        self.char_vocabulary_size = char_vocabulary_size

        config = BertConfig(
            vocab_size=char_vocabulary_size,
            is_decoder=True,
            add_cross_attention=True,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act="relu",
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            output_attentions=True)
        self.transformer = BertModel(config)

        if encoder is not None:
            self.transformer.embeddings = encoder.transformer.embeddings

        self.output_proj = nn.Linear(dim, char_vocabulary_size)
        self.output_proj.weight = \
            self.transformer.embeddings.word_embeddings.weight

    @property
    def embeddings(self) -> nn.Module:
        return self.transformer.embeddings.word_embeddings

    def _hidden_states(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            for_training: bool) -> Tuple[T, T, T]:
        batch_size = target_ids.size(0)
        to_prepend = torch.ones(
            (batch_size, 1),
            dtype=torch.int64).to(target_ids.device)

        if for_training:
            target_ids = target_ids[:, :-1]
            target_mask = target_mask[:, :-1]
        dec_input = torch.cat([to_prepend, target_ids], dim=1)
        input_mask = torch.cat([to_prepend, target_mask], dim=1)

        decoder_states, _, self_att, encdec_att = self.transformer(
            dec_input,
            attention_mask=input_mask,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask)

        return decoder_states, self_att, encdec_att

    def forward(
            self,
            encoder_states: T,
            encoder_mask: T,
            target_ids: T,
            target_mask: T,
            loss_function: nn.Module,
            log_details: bool = False) -> Tuple[T, Optional[Dict[str, Any]]]:

        decoder_states, self_att, encdec_att = self._hidden_states(
            encoder_states, encoder_mask, target_ids, target_mask,
            for_training=True)

        decoder_logits = self.output_proj(decoder_states)

        loss_per_char = loss_function(
            decoder_logits.reshape(-1, self.char_vocabulary_size),
            target_ids.reshape(-1))

        details = None
        if log_details:
            details = {}
            entropies = -(
                F.log_softmax(decoder_logits, dim=-1) *
                F.softmax(decoder_logits, dim=-1)).sum(2) * target_mask
            details["decoder_mask"] = target_mask
            details["output_entropy"] = (entropies).sum() / target_mask.sum()
            details["decoder_self_attention"] = self_att
            details["ecnode_decoder_attention"] = encdec_att

        return (
            (loss_per_char * target_mask.reshape(-1)).sum() / target_mask.sum(),
            details)

    @torch.no_grad()
    def greedy_decode(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            sample: bool = False,
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
            states, _, _ = self._hidden_states(
                encoder_states, encoder_mask,
                decoded,
                torch.ones_like(decoded, dtype=torch.float),
                for_training=False)
            last_state = states[:, -1]

            out_logits = self.output_proj(last_state)
            if sample:
                out_dist = F.softmax(out_logits, dim=1)
                new_char = torch.multinomial(out_dist, 1)
            else:
                new_char = out_logits.argmax(1, keepdim=True)
            finished = finished + (new_char == eos_token_id)

            decoded = torch.cat((decoded, new_char), dim=1)
            if finished.all():
                break

        return decoded, None

    @torch.no_grad()
    def beam_search(
            self,
            encoder_states: T,
            encoder_mask: T,
            eos_token_id: int,
            beam_size: int,
            len_norm: float,
            max_len: int = 300) -> Tuple[T, T]:
        batch_size = encoder_states.size(0)
        cur_len = 1
        current_beam = 1

        decoded = torch.ones(
            (batch_size, 1, 0),
            dtype=torch.long).to(encoder_states.device)
        finished = torch.zeros(
            (batch_size, 1, 0),
            dtype=torch.bool).to(encoder_states.device)
        scores = torch.zeros((batch_size, 1)).to(encoder_states.device)

        flat_decoded = decoded.squeeze(1)
        flat_finished = finished.squeeze(1)
        while cur_len < max_len:
            states, _, _ = self._hidden_states(
                encoder_states, encoder_mask,
                flat_decoded, ~flat_finished,
                for_training=False)
            last_state = states[:, -1:]
            next_token_logprobs = F.log_softmax(self.output_proj(
                last_state), dim=-1)

            # get scores of all expanded hypotheses
            candidate_scores = (
                scores.unsqueeze(2) +
                next_token_logprobs.reshape(batch_size, current_beam, -1))
            norm_factor = torch.pow(
                (1 - finished.float()).sum(2, keepdim=True) + 1, len_norm)
            normed_scores = candidate_scores / norm_factor

            # reshape for beam members and get top k
            _, best_indices = normed_scores.reshape(
                batch_size, -1).topk(beam_size, dim=-1)
            next_symbol_ids = best_indices % self.char_vocabulary_size
            hypothesis_ids = best_indices // self.char_vocabulary_size

            # numbering elements in the extended batch, i.e. beam size copies
            # of each batch element
            beam_offset = torch.arange(
                0, batch_size * current_beam, step=current_beam,
                dtype=torch.long, device=encoder_states.device)
            global_best_indices = (
                beam_offset.unsqueeze(1) + hypothesis_ids).reshape(-1)

            # now select appropriate histories
            decoded = torch.cat((
                flat_decoded.index_select(
                    0, global_best_indices).reshape(batch_size, beam_size, -1),
                next_symbol_ids.unsqueeze(-1)), dim=2)
            reordered_finished = flat_finished.index_select(
                0, global_best_indices).reshape(batch_size, beam_size, -1)
            finished_now = (next_symbol_ids == eos_token_id)
            if reordered_finished.size(2) > 0:
                finished_now += reordered_finished[:, :, -1]
            finished = torch.cat((
                reordered_finished,
                finished_now.unsqueeze(-1)), dim=2)
            if finished_now.all():
                break

            # re-order scores
            scores = candidate_scores.reshape(
                batch_size, -1).gather(-1, best_indices)

            # tile encoder after first step
            if cur_len == 1:
                encoder_states = encoder_states.unsqueeze(1).repeat(
                    1, beam_size, 1, 1).reshape(
                        batch_size * beam_size, encoder_states.size(1),
                        encoder_states.size(2))
                encoder_mask = encoder_mask.unsqueeze(1).repeat(
                    1, beam_size, 1).reshape(batch_size * beam_size, -1)

            # in the first iteration, beam size is 1, in the later ones,
            # it is the real beam size
            flat_decoded = decoded.reshape(-1, cur_len)
            flat_finished = finished.reshape(-1, cur_len)
            current_beam = beam_size
            cur_len += 1

        return (decoded[:, 0], finished[:, 0].logical_not())
