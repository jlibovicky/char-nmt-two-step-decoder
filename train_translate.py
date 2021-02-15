#!/usr/bin/env python3

import argparse
import logging
from typing import Iterable, Dict
import random
import sys

import sacrebleu
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

from char_tokenizer import from_data
from seq_to_seq import Seq2SeqModel
from lr_scheduler import NoamLR
from experiment import experiment_logging, get_timestamp


T = torch.Tensor


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def preprocess_data(train_src, train_tgt, batch_size, tokenizer=None):
    logging.info("Loading file %s.", train_src.name)
    src_text = [line.strip() for line in train_src]
    logging.info("Loading file %s.", train_tgt.name)
    tgt_text = [line.strip() for line in train_tgt]

    if tokenizer is None:
        logging.info("Initializing tokenizer.")
        tokenizer = from_data(src_text + tgt_text)

    batches = []
    src_batch, tgt_batch = [], []
    total_sentences = 0
    skipped = 0
    logging.info("Binarizing and batching data.")
    for src, tgt in zip(src_text, tgt_text):
        total_sentences += 1
        if len(src) > 300 or len(tgt) > 300:
            skipped += 1
            continue

        src_batch.append(src)
        tgt_batch.append(tgt)
        if len(src_batch) >= batch_size:
            batches.append((
                tokenizer.batch_encode_plus(src_batch),
                tokenizer.batch_encode_plus(tgt_batch)))
            src_batch, tgt_batch = [], []
    if src_batch:
        batches.append((
            tokenizer.batch_encode_plus(src_batch),
            tokenizer.batch_encode_plus(tgt_batch)))

    logging.info(
        "Skipped %d sentences from %d in the dataset.",
        skipped, total_sentences)
    return tokenizer, batches


@torch.no_grad()
def validate(model, batches, loss_function, device, tokenizer):
    model.eval()
    loss_sum = 0
    decoded_sentences = []
    target_sentences = []
    for i, ((src_data, src_mask), (tgt_data, tgt_mask)) in enumerate(batches):
        src_data, src_mask = src_data.to(device), src_mask.to(device)
        loss = model(
            src_data, src_mask,
            tgt_data.to(device), tgt_mask.to(device), loss_function)
        loss_sum += loss
        decoded_ids = model.greedy_decode(
            src_data, src_mask, tokenizer.eos_token_id)[0]
        decoded_sentences.extend(tokenizer.batch_decode(decoded_ids))
        target_sentences.extend(tokenizer.batch_decode(tgt_data))

        if i == 0:
            for hyp, ref in zip(decoded_sentences[:5], target_sentences):
                logging.info("'%s' -> '%s'", ref, hyp)

    model.train()

    bleu = sacrebleu.corpus_bleu(decoded_sentences, [target_sentences])

    val_samples = zip(target_sentences, decoded_sentences[:10])

    return loss_sum / len(batches), bleu.score, val_samples


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "train_src", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument(
        "train_tgt", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument(
        "val_src", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument(
        "val_tgt", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument(
        "test_src", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument(
        "test_src", type=argparse.FileType("r"), nargs="?", default=sys.stdin)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--char-emb-dim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--shrink-factor", type=int, default=5)
    parser.add_argument("--nar-output", default=False, action="store_true")
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--highway-layers", type=int, default=2)
    parser.add_argument("--char-ff-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--delay-update", type=int, default=1)
    parser.add_argument("--validation-period", type=int, default=40)
    parser.add_argument(
        "--convolutions", nargs="+",
        default=[200, 200, 250, 250, 300, 300, 300, 300],
        type=int)
    parser.add_argument(
        "--vanilla-decoder", action="store_true", default=False)
    parser.add_argument(
        "--share-char-repr", action="store_true", default=False)
    parser.add_argument("--name", default="experiment", type=str)
    args = parser.parse_args()

    experiment_dir = experiment_logging(
        "experiments", f"{args.name}_{get_timestamp()}", args)
    tb_writer = SummaryWriter(experiment_dir)

    logging.info("Load and binarize data.")
    tokenizer, train_batches = preprocess_data(
        args.train_src, args.train_tgt, args.batch_size)
    _, val_batches = preprocess_data(
        args.val_src, args.val_tgt, args.batch_size, tokenizer=tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Initializing model on device %s.", device)
    model = Seq2SeqModel(
        tokenizer.vocab_size,
        conv_filters=args.convolutions,
        char_embedding_dim=args.char_emb_dim,
        dim=args.dim,
        shrink_factor=args.shrink_factor,
        nar_output=args.nar_output,
        highway_layers=args.highway_layers,
        char_ff_layers=args.char_ff_layers,
        ff_dim=2 * args.dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        vanilla_decoder=args.vanilla_decoder,
        share_char_repr=args.share_char_repr).to(device)

    char_params = model.char_level_param_count
    logging.info(
        "Parameters in char processing layers %d, i.e. as "
        "%d word vocabulary.",
        char_params, char_params // args.dim)

    loss_function = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters())
    scheduler = NoamLR(optimizer, args.warmup)

    logging.info("Training starts.")
    steps = 0
    for epoch_n in range(args.epochs):
        logging.info("Epoch %d starts, so far %d steps.", epoch_n + 1, steps)
        for (src_data, src_mask), (tgt_data, tgt_mask) in train_batches:
            steps += 1

            loss = model(
                src_data.to(device), src_mask.to(device),
                tgt_data.to(device), tgt_mask.to(device), loss_function)

            loss.backward()

            if steps % args.validation_period == 0:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        tb_writer.add_histogram(name, param.grad, steps)

            if steps % args.delay_update == 0:
                logging.info("Step %d, loss %.4g", steps, loss.item())
                tb_writer.add_scalar("loss/train", loss, global_step=steps)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                torch.cuda.empty_cache()

            if steps % args.validation_period == 0:
                val_loss, val_bleu, val_samples = validate(
                    model, val_batches, loss_function, device, tokenizer)
                tb_writer.add_scalar("loss/val", val_loss, global_step=steps)
                tb_writer.add_scalar("bleu/val", val_bleu, global_step=steps)

                for i, (ref, hyp) in enumerate(val_samples):
                    tb_writer.add_text(f"{i + 1}", f"__ref:__ {ref}<br />__hyp:__ {hyp}", steps)

                logging.info(
                    "VALIDATION: Step %d (epoch %d), loss %.4g, BLEU: %.4g",
                    steps, epoch_n + 1, val_loss, val_bleu)
        random.shuffle(train_batches)

    tb_writer.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()
