#!/usr/bin/env python3

import argparse
import logging
import os
from typing import Iterable
import random
import sys

import joblib
import numpy as np
import yaml
import sacrebleu
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim

import char_tokenizer
import bigram_tokenizer
from seq_to_seq import Seq2SeqModel
from lr_scheduler import NoamLR
from experiment import experiment_logging, get_timestamp


T = torch.Tensor


logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def preprocess_data(
        train_src, train_tgt, batch_size, max_vocab_size: int,
        tokenizer_type: str,
        tokenizer=None, max_lines: int = 1000000):
    logging.info("Loading file %s.", train_src.name)
    src_text = [line.strip() for line in train_src]
    logging.info("Loading file %s.", train_tgt.name)
    tgt_text = [line.strip() for line in train_tgt]

    tokenizer_fn = None
    if tokenizer_type == "char":
        tokenizer_fn = char_tokenizer.from_data
    elif tokenizer_type == "bigram":
        tokenizer_fn = bigram_tokenizer.from_data

    if tokenizer is None:
        logging.info("Initializing tokenizer.")
        all_data = src_text + tgt_text
        random.shuffle(all_data)
        tokenizer = tokenizer_fn(
            all_data, max_vocab=max_vocab_size, max_lines=max_lines)

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


def cpu_save_state_dict(model, experiment_dir, name):
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, os.path.join(experiment_dir, name))


@torch.no_grad()
def validate(model, batches, loss_function, device, tokenizer, tb_writer,
             steps: int, log_details: bool):
    model.eval()
    loss_sum = 0
    source_sentences = []
    decoded_sentences = []
    target_sentences = []
    details_list = []
    for i, ((src_data, src_mask), (tgt_data, tgt_mask)) in enumerate(batches):
        src_data, src_mask = src_data.to(device), src_mask.to(device)
        loss, details = model(
            src_data, src_mask,
            tgt_data.to(device), tgt_mask.to(device), loss_function,
            log_details=log_details)
        details_list.append(details)
        loss_sum += loss
        decoded_ids = model.greedy_decode(
            src_data, src_mask, tokenizer.eos_token_id)[0]
        source_sentences.extend(tokenizer.batch_decode(src_data))
        decoded_sentences.extend(tokenizer.batch_decode(decoded_ids))
        target_sentences.extend(tokenizer.batch_decode(tgt_data))

        if i == 0:
            for hyp, ref in zip(decoded_sentences[:5], target_sentences):
                logging.info("'%s' -> '%s'", ref, hyp)

    model.train()

    bleu = sacrebleu.corpus_bleu(decoded_sentences, [target_sentences])
    chrf = sacrebleu.corpus_chrf(decoded_sentences, [target_sentences])

    val_samples = zip(
        source_sentences, target_sentences, decoded_sentences)
    val_loss = loss_sum / len(batches)

    tb_writer.add_scalar("loss/val", val_loss, global_step=steps)
    tb_writer.add_scalar("bleu/val", bleu.score, global_step=steps)
    tb_writer.add_scalar("chrf/val", chrf.score, global_step=steps)

    for i, (src, ref, hyp) in enumerate(val_samples):
        tb_writer.add_text(
            f"{i + 1}",
            f"`src:` {src}  \n"
            f"`ref:` {ref}  \n"
            f"`hyp:` {hyp}", steps)
        if i >= 9:
            break

    if log_details:
        output_entropy = np.mean([
            d["output_entropy"].cpu() for d in details_list])
        tb_writer.add_scalar(
            "details/output_entropy", output_entropy, global_step=steps)
        tb_writer.add_embedding(
            model.encoder.embeddings.weight,
            metadata=tokenizer.idx_to_str,
            global_step=steps,
            tag='Encoder embeddings')
        tb_writer.add_embedding(
            model.decoder.embeddings.weight[:len(tokenizer.idx_to_str)],
            metadata=tokenizer.idx_to_str,
            global_step=steps,
            tag='Decoder embeddings')
        encoder_self_att_entropies = [
            np.mean([d["enc_attention_entropies"][i].cpu()
                     for d in details_list])
            for i in range(model.encoder.layers)]
        decoder_self_att_entropies = [
            np.mean([d["dec_attention_entropies"][i].cpu()
                     for d in details_list])
            for i in range(model.encoder.layers)]
        encdec_att_entropies = [
            np.mean([d["encdec_attention_entropies"][i].cpu()
                     for d in details_list])
            for i in range(model.encoder.layers)]
        for i, ent in enumerate(encoder_self_att_entropies):
            tb_writer.add_scalar(
                f"details/encoder_self_att_entropy_layer{i + 1}",
                ent, global_step=steps)
        for i, ent in enumerate(decoder_self_att_entropies):
            tb_writer.add_scalar(
                f"details/decoder_self_att_entropy_layer{i + 1}",
                ent, global_step=steps)
        for i, ent in enumerate(encdec_att_entropies):
            tb_writer.add_scalar(
                f"details/encoder_decoder_att_entropy_layer{i + 1}",
                ent, global_step=steps)

    return val_loss, bleu.score, chrf.score, val_samples


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("tokenizer", type=str, help="File with saved tokenizer.")
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
        default=[128, 256, 512, 512, 512], type=int)
    parser.add_argument(
        "--vanilla-decoder", action="store_true", default=False)
    parser.add_argument(
        "--share-char-repr", action="store_true", default=False)
    parser.add_argument("--name", default="experiment", type=str)
    parser.add_argument(
        "--continue-training", type=str, default=None,
        help="Directory with experiment that should be continued. The model "
             "and the hyperparatemers get loaded from the directory, the "
             "datasets are taken from args.")
    args = parser.parse_args()

    if args.continue_training is not None:
        logging.info("Loading hyper-parameters from the previous experiement.")
        with open(os.path.join(args.continue_training, "args")) as f_args:
            previous_args = yaml.load(f_args)

        args.char_emb_dim = previous_args["char_emb_dim"]
        args.dim = previous_args["dim"]
        args.layers = previous_args["layers"]
        args.convolutions = previous_args["convolutions"]
        args.shrink_factor = previous_args["shrink_factor"]
        args.attention_heads = previous_args["attention_heads"]
        args.dropout = previous_args["dropout"]
        args.highway_layers = previous_args["highway_layers"]
        args.char_ff_layers = previous_args["char_ff_layers"]
        args.vanilla_decoder = previous_args["vanilla_decoder"]
        args.share_char_repr = previous_args["share_char_repr"]

        logging.info("Loading tokenizer from the previous experiement.")
        tokenizer = joblib.load(
            os.path.join(args.continue_training, "tokenizer.joblib"))
    else:
        tokenizer = None

    experiment_dir = experiment_logging(
        "experiments", f"{args.name}_{get_timestamp()}", args)
    tb_writer = SummaryWriter(experiment_dir)

    logging.info("Load and binarize data.")
    tokenizer, train_batches = preprocess_data(
        args.train_src, args.train_tgt, args.batch_size, args.max_vocab,
        args.tokenizer_type, tokenizer=tokenizer)
    _, val_batches = preprocess_data(
        args.val_src, args.val_tgt, args.batch_size, args.max_vocab,
        args.tokenizer_type, tokenizer=tokenizer)
    joblib.dump(tokenizer, os.path.join(experiment_dir, "tokenizer.joblib"))

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

    if args.continue_training is not None:
        logging.info("Load model paramters from file.")
        state_dict = torch.load(
            os.path.join(args.continue_training, 'best_bleu.pt'),
            map_location=device)
        model.load_state_dict(state_dict)

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
    best_bleu = 0.0
    for epoch_n in range(args.epochs):
        logging.info("Epoch %d starts, so far %d steps.", epoch_n + 1, steps)
        for (src_data, src_mask), (tgt_data, tgt_mask) in train_batches:
            steps += 1

            loss, _ = model(
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
                val_loss, val_bleu, val_chrf, val_samples = validate(
                    model, val_batches, loss_function, device, tokenizer,
                    tb_writer, steps,
                    steps % (4 * args.validation_period) == 0)

                val_out_path = os.path.join(experiment_dir, "validation.out")
                with open(val_out_path, "w") as f_val:
                    for _, _, hyp in val_samples:
                        print(hyp, file=f_val)

                logging.info(
                    "VALIDATION: Step %d (epoch %d), loss %.4g, "
                    "BLEU: %.4g chrF: %.4g",
                    steps, epoch_n + 1, val_loss, val_bleu, val_chrf)

                if val_bleu > best_bleu:
                    logging.info("New best BLEU, saving model.")
                    best_bleu = val_bleu
                    cpu_save_state_dict(model, experiment_dir, "best_bleu.pt")
                cpu_save_state_dict(
                    model, experiment_dir, "last_checkpoint.pt")
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(experiment_dir, "last_optimizer.pt"))

        random.shuffle(train_batches)

    tb_writer.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()
