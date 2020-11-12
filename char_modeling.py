"""Classes handling character-level input and outpu."""

from typing import Dict, Iterable, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import BertConfig, BertEncoder

from axe import axe_loss


T = torch.Tensor


def encode_str(
        sentence: str,
        vocab_dict: Dict[str, int]) -> Optional[torch.LongTensor]:
    if any(c not in vocab_dict for c in sentence):
        return None
    return torch.tensor(
        [vocab_dict[c] + 2 for c in list(sentence) + ["</s>"]], dtype=torch.int64)


def decode_str(logprobs: T, lengths: T, vocab: List[str]) -> Iterable[str]:
    for indices, length in zip(logprobs.argmax(2), lengths):
        word: List[str] = []
        was_blank = True
        for idx in indices[:int(length)]:
            if idx == 1:
                was_blank = True
                continue
            char_to_add = vocab[int(idx) - 2]
            if was_blank or char_to_add != word[-1]:
                word.append(char_to_add)
            was_blank = False
        yield "".join(word)


class CharCNN(nn.Module):
    def __init__(
            self, vocab_size: int, dim: int, final_window: int,
            final_stride: int, dropout: float = 0.1) -> None:
        super().__init__()

        # TODO convolutions before doing the final one

        #if final_stride < 2:
        #    raise ValueError(
        #        "The final stride needs to reduce the sequence length.")

        self.final_window = final_window
        self.final_stride = final_stride
        self.dim = dim

        self.embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(1024, dim)
        self.final_cnn = nn.Sequential(
            nn.Conv1d(dim, dim, final_window, final_stride),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.GroupNorm(1, dim))

        self.final_mask_shrink = nn.MaxPool1d(final_window, final_stride)

    def forward(
            self, data: torch.LongTensor,
            mask: T,
            decoder_pad: bool = False,
            decoder_inference: bool = False,
            batch_size: T = None) -> Tuple[T, T]:

        if decoder_pad:
            assert batch_size is not None
            to_prepend = torch.ones((
                batch_size, 2 * self.final_window),
                dtype=torch.int64).to(data.device)
            if data.size(1) > 2 * self.final_window and not decoder_inference:
                data = data[:, :-2 * self.final_window]
            else:
                to_prepend_mask = torch.ones((
                    batch_size, 2 * self.final_window)).to(data.device)
                mask = torch.cat([to_prepend_mask, mask], dim=1)
            #if decoder_inference:
            #    print(data.dtype)
            data = torch.cat([to_prepend, data], dim=1)

        indices = torch.arange(data.size(1)).unsqueeze(0).to(data.device)
        position_embeddings = self.position_embeddings(indices)

        x = self.embeddings(data) + position_embeddings
        x = self.final_cnn(x.transpose(2, 1)).transpose(2, 1)

        out_mask = self.final_mask_shrink(mask.unsqueeze(1)).squeeze(1)

        return x, out_mask


class CharDecode(nn.Module):
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
            representation: T,
            encoder_mask: T,
            targets: torch.LongTensor = None,
            target_mask: T = None) -> Tuple[T, T, Optional[T]]:
        assert ((targets is None and target_mask is None) or
                (targets is not None and target_mask is not None))

        seq_lengths = encoder_mask.int().sum(1)

        logprobs = self.expand_layer(
            representation.transpose(2, 1)).transpose(2, 1)

        out_lenghts = self.final_stride * (seq_lengths - 1)
        loss = None
        if targets is not None and target_mask is not None:
            #loss = axe_loss(
            #    logits=logits,
            #    logit_lengths=out_lenghts,
            #    targets=targets,
            #    target_lengths=target_mask.int().sum(1),
            #    blank_index=1,
            #    delta=2.0)
            loss = self.loss_function(
                logprobs.transpose(0, 1), targets,
                input_lengths=out_lenghts,
                target_lengths=target_mask.int().sum(1))

        return logprobs, out_lenghts, loss


class Encoder(nn.Module):
    def __init__(
            self, vocab_size: int, dim: int, final_window: int,
            final_stride: int, dropout: float = 0.1,
            layers: int = 6, ff_dim: int = None, attention_heads: int = 8) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers

        self.char_encoder = CharCNN(
            vocab_size, dim, final_window, final_stride)

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

    def forward(self, data: T, mask: T) -> Tuple[T, T]:
        data, mask = self.char_encoder(data, mask)

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        data = self.transformer(
            data, attention_mask=extended_attention_mask,
            head_mask=[None] * self.layers)[0]

        return data, mask


class Decoder(nn.Module):
    def __init__(
            self, vocab_size: int, dim: int, final_window: int,
            final_stride: int, dropout: float = 0.1,
            layers: int = 6, ff_dim: int = None, attention_heads: int = 8,
            reuse_char_cnn: CharCNN = None) -> None:
        super().__init__()

        self.dim = dim
        self.ff_dim = ff_dim if ff_dim is not None else 2 * dim
        self.layers = layers

        if reuse_char_cnn is None:
            self.char_encoder = CharCNN(
                vocab_size, dim, final_window, final_stride)
        else:
            self.char_encoder = reuse_char_cnn

        config = BertConfig(
            vocab_size=vocab_size,
            is_decoder=True,
            hidden_size=dim,
            num_hidden_layers=layers,
            num_attention_heads=attention_heads,
            intermediate_size=self.ff_dim,
            hidden_act='relu',
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout)
        self.transformer = BertEncoder(config)

        self.char_output = CharDecode(
            vocab_size, dim, final_window, final_stride)

    def forward(
            self, encoder_states: T, encoder_mask: T,
            decoder_input: T, mask: T,
            inference_mode: bool = False,
            compute_loss: bool = True) -> Tuple[T, T, Optional[T]]:

        decoder_embeddings, mask = self.char_encoder(
            decoder_input, mask, decoder_pad=True,
            decoder_inference=inference_mode,
            batch_size=decoder_input.size(0))

        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        extended_encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        extended_encoder_mask = (1.0 - extended_encoder_mask) * -10000.0

        decoder_states = self.transformer(
            decoder_embeddings,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=extended_encoder_mask,
            head_mask=[None] * self.layers)[0]

        targets, target_mask = None, None
        if compute_loss:
            targets, target_mask = decoder_input, mask

        logprobs, out_lengths, loss = self.char_output(
            decoder_states, mask, targets, target_mask)

        return logprobs, out_lengths, loss


class Seq2SeqModel(nn.Module):

    def __init__(
            self, vocab_size: int, dim: int,
            encoder_window: int, encoder_stride: int,
            decoder_window: int, decoder_stride: int,
            dropout: float = 0.1,
            layers: int = 6, ff_dim: int = None, attention_heads: int = 8,
            reuse_char_cnn: CharCNN = None) -> None:
        super().__init__()

        self.encoder = Encoder(
            vocab_size, dim, encoder_window, encoder_stride, dropout,
            layers, ff_dim, attention_heads)
        self.decoder = Decoder(
            vocab_size, dim, decoder_window, decoder_stride, dropout,
            layers, ff_dim, attention_heads,
            reuse_char_cnn=self.encoder.char_encoder)

    def forward(
            self, src_batch: T, src_mask: T, tgt_batch: T, tgt_mask: T,
            compute_loss: bool = True, inference_mode: bool = False):
        encoded, enc_mask = self.encoder(src_batch, src_mask)

        logprobs, out_lengths, loss = self.decoder(
            encoded, enc_mask, tgt_batch, tgt_mask,
            compute_loss=compute_loss,
            inference_mode=inference_mode)

        return logprobs, out_lengths, loss

    @torch.no_grad()
    def greedy_decode(self, src_batch, max_len=300):
        input_mask = (src_batch != 0).float()
        encoder_states, encoded_mask = self.encoder(src_batch, input_mask)
        batch_size = src_batch.size(0)

        decoded = torch.zeros(
            (batch_size, 0), dtype=torch.int64).to(src_batch.device)

        for _ in range(max_len):
            logprobs, out_lenghts, _ = self.decoder(
                encoder_states, encoded_mask,
                decoded, (decoded != 0).float(),
                compute_loss=False, inference_mode=True)

            # finished = []
            decoded_as_list = []
            for sent_idx in logprobs.argmax(2):
                decoded_sent = []
                # TODO some policy about neighboring
                for char_id in sent_idx:
                    if char_id == 1:
                        continue
                    if char_id == 2:
                        break
                    decoded_sent.append(char_id)
                decoded_as_list.append(torch.tensor(decoded_sent))
            decoded = pad_sequence(
                decoded_as_list, batch_first=True).to(src_batch.device).long()

            #if all(finished_now):
            #    break

        return decoded
