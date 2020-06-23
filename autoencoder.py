#!/usr/bin/env python3

"""Autoencoder, a proof-of-concept test of char-level encoding and decoding."""

import argparse
import logging
from typing import List, Optional, Tuple
import sys

import editdistance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from char_modeling import CharCNN, CharCTCDecode


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


VOCAB = list(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?! ()[]")
VOCAB_DICT = {c: i for i, c in enumerate(VOCAB)}


def encode(sentence: str) -> Optional[torch.LongTensor]:
    if any(c not in VOCAB_DICT for c in sentence):
        return None
    return torch.tensor([VOCAB_DICT[c] + 2 for c in sentence])


def decode(logprobs: torch.Tensor, lengths: torch.Tensor) -> List[str]:
    for indices, length in zip(logprobs.argmax(2), lengths):
        word = []
        for idx in indices[:length]:
            if idx == 1:
                continue
            word.append(VOCAB[idx - 2])
        yield "".join(word)


class AutoEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, final_window: int, final_stride: int) -> None:
        super().__init__()

        self.encoder = CharCNN(vocab_size, dim, final_window, final_stride)
        self.decoder = CharCTCDecode(
            vocab_size, dim, final_window, 2 * final_stride)

    def forward(self, data: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (data != 0).float()
        encoded, enc_mask = self.encoder(data, mask)
        decoded, lenghts, loss = self.decoder(encoded, enc_mask, data, mask)

        return decoded, lenghts, loss


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "data", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--final-window", type=int, default=5)
    parser.add_argument("--final-stride", type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Initializing model.")
    model = AutoEncoder(
        len(VOCAB) + 2, args.dim,
        args.final_window, args.final_stride).to(device)

    optimizer = optim.Adam(model.parameters())

    logging.info("Pre-loading validation data.")
    val_sentences = []
    val_sentences_str = []
    while len(val_sentences) < 512:
        sentence = args.data.readline().strip()
        if len(sentence) > 512:
            continue
        encoded_str = encode(sentence)
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
        encoded_str = encode(sentence.strip())
        if encoded_str is not None:
            train_batch.append(encoded_str)

        if len(train_batch) < args.batch_size:
            continue

        train_tensor = pad_sequence(
            train_batch, batch_first=True).to(device)
        train_batch = []
        steps += 1

        _, _, loss = model(train_tensor)
        logging.info("Step %d, loss %.3f", steps, loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

        if steps % 20 == 0:
            with torch.no_grad():
                model.eval()
                val_decoded, val_lengths, val_loss = model(val_batch)
                model.train()

            decoded = list(decode(val_decoded, val_lengths))
            edit_dists = 0
            for hyp, ref in zip(decoded, val_sentences_str):
                edit_dists += editdistance.eval(hyp, ref) / len(ref)
            for hyp, ref in zip(decoded, val_sentences_str[:5]):
                logging.info("%s -> %s", ref, hyp)
            logging.info(
                    "VALIDATION: Step %d, loss %.4g, edit distance: %.4g",
                    steps, val_loss.item(), edit_dists / len(val_sentences))


if __name__ == "__main__":
    main()
