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

from char_modeling import Seq2SeqModel, encode_str, decode_str
from lr_scheduler import NoamLR


T = torch.Tensor


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


VOCAB = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?! ()[]") + ["</s>"]
VOCAB_DICT = {c: i for i, c in enumerate(VOCAB)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "data", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--final-window", type=int, default=5)
    parser.add_argument("--final-stride", type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Initializing model on device %s.", device)

    model = Seq2SeqModel(
        len(VOCAB) + 2, args.dim,
        args.final_window, args.final_stride,
        args.final_window, 2 * args.final_stride,
        layers=args.layers).to(device)

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
        _, _, loss = model(train_tensor, mask, train_tensor, mask)
        logging.info("Step %d, loss %.4g", steps, loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.cuda.empty_cache()

        if steps % 20 == 0:
            with torch.no_grad():
                model.eval()
                val_decoded = model.greedy_decode(val_batch)
                model.train()

            decoded = []
            for output in val_decoded:
                out_sent = []
                for char_id in output:
                    if char_id == 2:
                        break
                    out_sent.append(VOCAB[char_id - 2])
                decoded.append("".join(out_sent))

            edit_dists = 0
            for hyp, ref in zip(decoded, val_sentences_str):
                edit_dists += editdistance.eval(hyp, ref) / len(ref)
            for hyp, ref in zip(decoded, val_sentences_str[:5]):
                logging.info("%s -> %s", ref, hyp)
            logging.info(
                "VALIDATION: Step %d, loss %.4g, edit distance: %.4g",
                steps, 0, edit_dists / len(val_sentences))


if __name__ == "__main__":
    main()
