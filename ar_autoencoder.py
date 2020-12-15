#!/usr/bin/env python3

"""Autoencoder, a proof-of-concept test of char-level encoding and decoding."""

import argparse
import logging
from typing import Iterable, Dict, List, Optional, Tuple
import sys

import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_bert import BertConfig, BertEncoder

from char_modeling import encode_str
from lee_encoder import Encoder
from decoder import Decoder, VanillaDecoder
from lr_scheduler import NoamLR


T = torch.Tensor


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


VOCAB = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?! ()[]") + ["</s>"]
VOCAB_DICT = {c: i for i, c in enumerate(VOCAB)}


class Seq2SeqModel(nn.Module):
    def __init__(
            self, vocab_size: int,
            conv_filters: List[int],
            char_embedding_dim: int = 128,
            dim: int = 512,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            ff_dim: int = None,
            layers: int = 6,
            attention_heads: int = 8,
            dropout: float = 0.1,
            vanilla_decoder: bool = False) -> None:
        super().__init__()

        self.layers = layers

        self.encoder = Encoder(
            vocab_size, char_embedding_dim, conv_filters, dim, shrink_factor, highway_layers,
            ff_dim, layers, attention_heads, dropout)
        if vanilla_decoder:
            self.decoder = VanillaDecoder(
                vocab_size, dim, layers, ff_dim, attention_heads, dropout)
        else:
            self.decoder = Decoder(
                vocab_size, char_embedding_dim, dim, shrink_factor, highway_layers,
                layers, ff_dim, attention_heads, dropout)

    def forward(
            self, src_batch: T, src_mask: T, tgt_batch: T, tgt_mask: T,
            loss_function: nn.Module) -> T:
        encoded, enc_mask = self.encoder(src_batch, src_mask)
        loss = self.decoder(
            encoded, enc_mask, tgt_batch, tgt_mask, loss_function)

        return loss

    @torch.no_grad()
    def greedy_decode(self, src_batch, max_len=300):
        input_mask = (src_batch != 0).float()
        encoder_states, encoded_mask = self.encoder(src_batch, input_mask)
        decoded, mask = self.decoder.greedy_decode(encoder_states, encoded_mask)

        return decoded, mask


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "data", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--char-emb-dim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--shrink-factor", type=int, default=5)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--highway-layers", type=int, default=2)
    parser.add_argument("--convolutions", nargs="+", default=[200, 200, 250, 250, 300, 300, 300, 300], type=int)
    parser.add_argument("--vanilla-decoder", action="store_true", default="False")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Initializing model on device %s.", device)

    model = Seq2SeqModel(
        len(VOCAB) + 2,
        conv_filters=args.convolutions,
        char_embedding_dim=args.char_emb_dim,
        dim=args.dim,
        shrink_factor=args.shrink_factor,
        highway_layers=args.highway_layers,
        ff_dim=2 * args.dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        vanilla_decoder=args.vanilla_decoder).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = NoamLR(optimizer, 1000)

    logging.info("Pre-loading validation data.")
    val_sentences = []
    val_sentences_str = []
    while len(val_sentences) < args.batch_size:
        sentence = args.data.readline().strip()
        if len(sentence) > 512:
            continue
        encoded_str = encode_str(sentence, VOCAB_DICT)
        if encoded_str is not None:
            val_sentences.append(encoded_str)
            val_sentences_str.append(sentence)
    val_batch = pad_sequence(val_sentences, batch_first=True).to(device)

    logging.info("Training starts.")
    train_batch = []
    steps = 0
    for sentence in args.data:
        if len(sentence) > 512:
            continue
        encoded_str = encode_str(sentence.strip(), VOCAB_DICT)
        if encoded_str is not None:
            train_batch.append(encoded_str)

        if len(train_batch) < args.batch_size:
            continue

        train_tensor = pad_sequence(
            train_batch, batch_first=True).to(device)
        train_batch = []
        steps += 1

        mask = (train_tensor != 0).float()
        loss = model(train_tensor, mask, train_tensor, mask, loss_function)
        logging.info("Step %d, loss %.4g", steps, loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.cuda.empty_cache()

        if steps % 40 == 0:
            with torch.no_grad():
                model.eval()
                val_mask = (val_batch != 0).float()
                val_loss = model(
                    val_batch, val_mask, val_batch, val_mask, loss_function)
                val_decoded = model.greedy_decode(val_batch)[0]
                model.train()

            decoded = []
            for output in val_decoded:
                out_sent = []
                for char_id in output:
                    if VOCAB[char_id - 2] == "</s>":
                        break
                    out_sent.append(VOCAB[char_id - 2])
                decoded.append("".join(out_sent))

            edit_dists = 0
            for hyp, ref in zip(decoded, val_sentences_str):
                edit_dists += editdistance.eval(hyp, ref) / len(ref)
            for hyp, ref in zip(decoded, val_sentences_str[:5]):
                logging.info("'%s' -> '%s'", ref, hyp)
            logging.info(
                "VALIDATION: Step %d, loss %.4g, edit distance: %.4g",
                steps, val_loss, edit_dists / len(val_sentences))


if __name__ == "__main__":
    main()
