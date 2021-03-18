#!/usr/bin/env python3

import argparse
import logging
import random

import joblib

import char_tokenizer
import bigram_tokenizer
import bpe_tokenizer


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "train_src", type=argparse.FileType("r"))
    parser.add_argument(
        "train_tgt", type=argparse.FileType("r"))
    parser.add_argument(
        "output", type=argparse.FileType("wb"))
    parser.add_argument("--max-vocab-size", type=int, default=300)
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument(
        "--tokenizer-type", type=str, default="char",
        choices=["char", "bigram", "bpe"])
    args = parser.parse_args()

    logging.info("Loading file %s.", args.train_src.name)
    src_text = [line.strip() for line in args.train_src]
    args.train_src.close()
    logging.info("Loading file %s.", args.train_tgt.name)
    tgt_text = [line.strip() for line in args.train_tgt]
    args.train_tgt.close()

    tokenizer_fn = None
    if args.tokenizer_type == "char":
        tokenizer_fn = char_tokenizer.from_data
    elif args.tokenizer_type == "bigram":
        tokenizer_fn = bigram_tokenizer.from_data
    elif args.tokenizer_type == "bpe":
        tokenizer_fn = bpe_tokenizer.from_data

    logging.info("Initializing tokenizer.")
    all_data = src_text + tgt_text
    random.shuffle(all_data)
    tokenizer = tokenizer_fn(
        all_data,
        max_vocab=args.max_vocab_size,
        max_lines=args.max_lines)

    joblib.dump(tokenizer, args.output)
    logging.info("Tokenzier saved to %s.", args.output.name)


if __name__ == "__main__":
    main()
