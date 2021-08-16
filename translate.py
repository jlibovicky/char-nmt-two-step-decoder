#!/usr/bin/env python3

"""Translated using a trained model."""

import argparse
import logging
import os
import sys

import joblib # type: ignore
import yaml
import torch

from seq_to_seq import Seq2SeqModel


T = torch.Tensor


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "model_dir", type=str, help="Directory with a trained model.")
    parser.add_argument(
        "input", nargs="?", default=sys.stdin, type=argparse.FileType("r"),
        help="Input as plain text.")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--len-norm", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--skip-pretokenization", action="store_true", default=False)
    parser.add_argument(
        "--force-char-segmentation", action="store_true", default=False)
    args = parser.parse_args()

    logging.info("Loading tokenizer.")
    tokenizer = joblib.load(
        os.path.join(args.model_dir, "tokenizer.joblib"))

    if args.skip_pretokenization:
        tokenizer.pretokenization = "skip"
    if args.force_char_segmentation:
        tokenizer.pretokenization = "char"

    with open(os.path.join(args.model_dir, "args")) as f_args:
        exp_args = yaml.load(f_args, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Initializing model on device %s.", device)
    model = Seq2SeqModel(
        tokenizer.vocab_size,
        conv_filters=exp_args.get("convolutions"),
        char_embedding_dim=exp_args.get("char_emb_dim"),
        dim=exp_args.get("dim"),
        shrink_factor=exp_args.get("shrink_factor"),
        nar_output=exp_args.get("nar_output"),
        highway_layers=exp_args.get("highway_layers"),
        char_ff_layers=exp_args.get("char_ff_layers"),
        ff_dim=2 * exp_args.get("dim"),
        layers=exp_args.get("layers"),
        attention_heads=exp_args.get("attention_heads"),
        dropout=exp_args.get("dropout"),
        vanilla_encoder=exp_args.get("vanilla_encoder"),
        vanilla_decoder=exp_args.get("vanilla_decoder"),
        share_char_repr=exp_args.get("share_char_repr")).to(device)
    model.eval()

    logging.info("Load model parameters from file.")

    ckpt_file = os.path.join(args.model_dir, "best_bleu.pt")
    if not os.path.exists(ckpt_file):
        ckpt_file = os.path.join(args.model_dir, "best_chrf.pt")

    state_dict = torch.load(
        ckpt_file, map_location=device)
    model.load_state_dict(state_dict)
    logging.info("Translating.")

    def decode_batch(input_batch):
        src_data, src_mask = tokenizer.batch_encode_plus(input_batch)
        if args.beam_size == 1:
            decoded_ids = model.greedy_decode(
                src_data.to(device), src_mask.to(device),
                tokenizer.eos_token_id)[0]
        else:
            decoded_ids = model.beam_search(
                src_data.to(device), src_mask.to(device),
                tokenizer.eos_token_id,
                beam_size=args.beam_size, len_norm=args.len_norm)[0]
        for output in tokenizer.batch_decode(decoded_ids):
            print(output)

    current_batch = []
    for line in args.input:
        current_batch.append(line.strip())
        if len(current_batch) > args.batch_size:
            decode_batch(current_batch)
            current_batch = []
    if current_batch:
        decode_batch(current_batch)

    logging.info("Done.")


if __name__ == "__main__":
    main()
