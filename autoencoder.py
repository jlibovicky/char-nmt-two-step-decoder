#!/usr/bin/env python3

"""Train autoencoder for latent variables. """

from typing import IO, List, Tuple, Union

import argparse
from dataclasses import dataclass
import logging
import random

import editdistance
from fairseq.models.lightconv import LightConvEncoderLayer
import joblib
import sacrebleu
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from encoder import Encoder
from perceiver import PerceiverEncoder
from experiment import experiment_logging, get_timestamp
from char_tokenizer import BaseTokenizer
from lr_scheduler import NoamLR

T = torch.Tensor

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


@dataclass
class DynamicConvArgs:
    """This class simulates the FairSeq arguments."""
    encoder_embed_dim: int = 512
    encoder_conv_type = "dynamic"
    encoder_glu: bool = True
    weight_softmax: bool = True
    encoder_attention_heads: int = 8
    weight_dropout: float = 0.1
    dropout: float = 0.1
    relu_dropout: float = 0.1
    input_dropout: float = 0.1
    encoder_normalize_before: bool = False

    @property
    def encoder_conv_dim(self) -> int:
        return self.encoder_embed_dim

    @property
    def encoder_ffn_embed_dim(self) -> int:
        return 2 * self.encoder_embed_dim


class ConvDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            vocab_size: int,
            latent_vocab_size: int,
            shrink_factor: int,
            layers: int = 3,
            window: int = 3,
            dropout: float=0.1,
            max_len: int=1000,
            use_dynamic_convolutions: bool=False):
        super().__init__()

        self.max_len = max_len
        self.layers = layers
        self.shrink_factor = shrink_factor
        self.use_dynamic_convolutions = use_dynamic_convolutions
        self.embedding_dim = embedding_dim
        self.latent_embedding = nn.Linear(
            latent_vocab_size, 2 * shrink_factor * embedding_dim)

        self.position_embeddings = nn.Parameter(
            torch.randn((1, max_len, embedding_dim)))
        self.position_embeddings.requires_grad = True

        self.pre_deconv = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout))

        self.cnn_layers = nn.ModuleList()
        self.cnn_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(layers):
            if use_dynamic_convolutions:
                self.cnn_layers.append(
                    LightConvEncoderLayer(
                        DynamicConvArgs(encoder_embed_dim=embedding_dim,
                                        weight_dropout=dropout,
                                        dropout=dropout),
                        kernel_size=window))
            else:
                self.cnn_layers.append(
                    nn.Conv1d(embedding_dim, 2 * embedding_dim,
                              window, padding=window // 2))
                self.cnn_norms.append(nn.LayerNorm(embedding_dim))

        self.output_proj = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, vocab_size + 1))

    def forward(self, latent_words: T, mask: T) -> Tuple[T, T]:
        batch_size = latent_words.size(0)
        output_mask = mask.unsqueeze(2).repeat(
            1, 1, 2 * self.shrink_factor).reshape(batch_size, -1)
        bool_mask = output_mask.bool()

        latent_embedded = self.latent_embedding(
            latent_words).reshape((batch_size, -1, self.embedding_dim))
        # Snippet to check this works
        # t = torch.tensor([[[1,1,2,2,3,3], [4,4,5,5,6,6]]])
        # t.reshape(1, -1, 2) returns what it should return

        if latent_embedded.size(1) > self.max_len:
            raise ValueError("Too long input.")
        latent_embedded = (
            latent_embedded +
            self.position_embeddings[:, :latent_embedded.size(1)])
        latent_embedded = self.pre_deconv(latent_embedded)

        output = latent_embedded
        for i in range(self.layers):
            if self.use_dynamic_convolutions:
                output = self.cnn_layers[i](
                    output.transpose(0, 1), bool_mask).transpose(0, 1)
            else:
                cnn_output = output * output_mask.unsqueeze(2)
                cnn_output = F.glu(self.cnn_layers[i](
                    output.transpose(2, 1)).transpose(2, 1))
                output = self.cnn_norms[i](self.dropout(cnn_output) + output)

        logits = self.output_proj(output)

        return logits, output_mask


class DeconvDecoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            vocab_size: int,
            latent_vocab_size: int,
            layers: int = 3,
            window: int = 3,
            stride: int = 2,
            dropout: float=0.1):
        super().__init__()

        self.layers = layers
        self.embedding_dim = embedding_dim
        self.latent_embedding = nn.Linear(
            latent_vocab_size, embedding_dim)
        self.stride = stride

        self.pre_deconv = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim))

        self.cnn_layers = nn.ModuleList()
        self.cnn_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for _ in range(layers):
            self.cnn_layers.append(
                nn.ConvTranspose1d(
                    embedding_dim, 2 * embedding_dim,
                    window, stride=stride))
            self.cnn_norms.append(nn.LayerNorm(embedding_dim))

        self.output_proj = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, vocab_size + 1))

    def forward(self, latent_words: T, mask: T) -> Tuple[T, T]:
        batch_size = latent_words.size(0)
        output_mask = mask

        latent_embedded = self.latent_embedding(
            latent_words).reshape((batch_size, -1, self.embedding_dim))
        latent_embedded = self.pre_deconv(latent_embedded)

        output = latent_embedded
        for i in range(self.layers):
            # TODO solve masking somehow
            #cnn_output = output * output_mask.unsqueeze(2)
            cnn_output = F.glu(self.cnn_layers[i](
                output.transpose(2, 1)).transpose(2, 1))
            #output = self.cnn_norms[i](self.dropout(cnn_output) + output)
            output = self.cnn_norms[i](self.dropout(cnn_output))
            output_mask = output_mask.unsqueeze(1).repeat(
                1, self.stride, 1).reshape(batch_size, -1)

        logits = self.output_proj(output)

        return logits, output_mask


class AutoEncoder(nn.Module):
    def __init__(
            self,
            encoder_type: str,
            decoder: Union[ConvDecoder, DeconvDecoder],
            vocab_size: int,
            conv_filters: List[int],
            latent_vocab_size: int = 8000,
            char_embedding_dim: int = 128,
            dim: int = 512,
            shrink_factor: int = 5,
            highway_layers: int = 2,
            char_ff_layers: int = 2,
            ff_dim: int = None,
            layers: int = 6,
            attention_heads: int = 8,
            dropout: float = 0.1,
            temperature: float = 2.0
            ) -> None:
        super().__init__()

        self.char_embedding_dim = char_embedding_dim
        self.temperature = temperature

        if encoder_type == "cnn":
            self.encoder = Encoder(
                vocab_size=vocab_size,
                char_embedding_dim=char_embedding_dim,
                conv_filters=conv_filters,
                intermediate_cnn_layers=3,
                dim=dim,
                shrink_factor=shrink_factor,
                highway_layers=highway_layers,
                char_ff_layers=char_ff_layers,
                ff_dim=ff_dim, layers=layers,
                attention_heads=attention_heads,
                dropout=dropout,
                decoder_style_padding=False)
        elif encoder_type == "perceiver":
            self.encoder = PerceiverEncoder(
                vocab_size=vocab_size,
                char_emb_dim=char_embedding_dim,
                shrink_factor=shrink_factor,
                transformer_dim=dim,
                layers=layers,
                dropout=dropout,
                attention_heads=attention_heads)
        else:
            raise ValueError(f"Unkown encoder type: {encoder_type}")

        self.latent_projection = nn.Linear(dim, latent_vocab_size)
        self.decoder = decoder

    def forward(
            self, data: T, mask: T, hard_gumbel: bool=False,
            inference: bool=False) -> Tuple[T, T]:
        encoded, latent_mask, _ = self.encoder(data, mask)

        latent_logits = self.latent_projection(encoded)
        if inference:
            latent_words = F.one_hot(
                latent_logits.argmax(2),
                num_classes=latent_logits.size(2)).float()
        else:
            latent_words = torch.nn.functional.gumbel_softmax(
                latent_logits,
                tau=self.temperature, hard=hard_gumbel, dim=-1)

        logits, output_mask = self.decoder(latent_words, latent_mask)
        return logits, output_mask


def preprocess_data(
        tokenizer: BaseTokenizer,
        train_src: IO[str],
        batch_size: int,
        sort_by_length: bool) -> List[Tuple[Tuple[T, T], Tuple[T, T]]]:
    logging.info("Loading file '%s'.", train_src.name)
    src_text = [line.strip() for line in train_src]

    binarized = _binarize_data(tokenizer, src_text, sort_by_length)
    batched = _batch_data(binarized, batch_size, tokenizer.pad_token_id)

    return batched


def _binarize_data(
        tokenizer: BaseTokenizer,
        src_text: List[str],
        sort_by_length: bool) -> List[T]:
    logging.info("Binarizing data.")

    total_sentences = 0
    skipped = 0
    binarized: List[T] = []

    pbar = trange(len(src_text), unit="sentences")
    for _, src in zip(pbar, src_text):
        total_sentences += 1
        if len(src) > 300:
            skipped += 1
            continue

        binarized.append( # type: ignore
            tokenizer.batch_encode_plus(
                [src], return_attention_mask=False)[0])

    logging.info(
        "Skipped %d sentences from %d in the dataset.",
        skipped, total_sentences)

    if sort_by_length:
        logging.info("Sorting data by length.")
        binarized.sort(key=lambda p: p.size(0))

    return binarized


def _batch_data(
        binarized: List[T], max_size: int,
        pad_idx: int) -> List[Tuple[T, T]]:
    batches = []
    max_cur_src_len = 0

    src_batch: List[T] = []

    logging.info("Batching data.")
    def process_batch() -> None:
        src_pt = pad_sequence(
            src_batch, batch_first=True, padding_value=pad_idx)
        src_mask = (src_pt != pad_idx).float()
        batches.append((src_pt, src_mask))

    pbar = trange(len(binarized), unit="sentences")
    for _, src in zip(pbar, binarized):
        if src.size(0) > max_size:
            raise ValueError(
                "Single instance is longer than maximum batch size.")
        future_max_src_len = max(max_cur_src_len, src.size(0))
        if len(src_batch) * future_max_src_len > max_size:
            process_batch()
            src_batch = []
            max_cur_src_len = 0
        src_batch.append(src)
        max_cur_src_len = future_max_src_len

    process_batch()
    return batches


@torch.no_grad()
def validate(model: AutoEncoder, batches: List[Tuple[T, T]],
             loss_function, device, tokenizer, hard_gumbel=True, inference=True):
    model.eval()
    source_sentences = []
    decoded_sentences = []
    for i, (src_data, src_mask) in enumerate(batches):
        src_data, src_mask = src_data.to(device), src_mask.to(device)
        decoded_logits, output_mask = model(
            src_data, src_mask, hard_gumbel=hard_gumbel, inference=inference)

        loss = loss_function(
            F.log_softmax(decoded_logits.transpose(0, 1), dim=-1),
            src_data + 1,
            output_mask.sum(1).int(),
            src_mask.sum(1).int())

        decoded_ids = (decoded_logits.argmax(2) - 1) * output_mask.int()
        source_sentences.extend(tokenizer.batch_decode(src_data))
        decoded_ids[decoded_ids == -1] = 0
        decoded_sentences.extend(tokenizer.batch_decode(decoded_ids))

        if i == 0:
            for hyp, ref in zip(decoded_sentences[:5], source_sentences):
                logging.info("'%s' -> '%s'", ref, hyp)

    model.train()

    acc_sum = 0.
    edit_sum = 0.
    for ref, hyp in zip(source_sentences, decoded_sentences):
        acc_sum += float(hyp == ref)
        edit_sum += editdistance.distance(hyp, ref) / len(ref)
    bleu = sacrebleu.corpus_bleu(decoded_sentences, [source_sentences])

    return (
        acc_sum / len(source_sentences),
        edit_sum / len(source_sentences),
        bleu.score,
        loss)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "tokenizer", type=str, help="File with saved tokenizer.")
    parser.add_argument(
        "train", type=argparse.FileType("r"))
    parser.add_argument(
        "val", type=argparse.FileType("r"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--char-emb-dim", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--shrink-factor", type=int, default=5)
    parser.add_argument(
        "--encoder-type", type=str, default="cnn",
        choices=["cnn", "perceiver"])
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--highway-layers", type=int, default=0)
    parser.add_argument("--char-ff-layers", type=int, default=0)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument(
        "--decoder-dynamic-convolutions", action="store_true",
        default=False)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--warmup", type=int, default=4000)
    parser.add_argument("--delay-update", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.7)
    parser.add_argument("--validation-period", type=int, default=1000)
    parser.add_argument(
        "--convolutions", nargs="+",
        default=[250, 250, 250, 250], type=int)
    parser.add_argument("--latent-vocab-size", type=int, default=32000)
    parser.add_argument("--latent-emb-dim", type=int, default=256)
    parser.add_argument(
        "--experiment-dir", type=str, default="autoencoder_search")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--soft-gumble-steps", type=int, default=1000,
        help="Number of steps with soft distribution.")
    parser.add_argument(
        "--min-temperature", type=float, default=0.5)
    parser.add_argument(
        "--temperature-decrease-factor", type=float, default=0.99999)
    args = parser.parse_args()

    experiment_dir = experiment_logging(
        args.experiment_dir, f"{get_timestamp()}", args)
    tb_writer = SummaryWriter(experiment_dir)

    logging.info("Loading tokenizer from '%s'.", args.tokenizer)
    tokenizer = joblib.load(args.tokenizer)
    tokenizer.ctc_mode = True

    logging.info("Load and binarize data.")
    train_batches = preprocess_data(
        tokenizer, args.train,
        args.batch_size, sort_by_length=True)
    random.shuffle(train_batches)
    val_batches = preprocess_data(
        tokenizer, args.val,
        args.batch_size, sort_by_length=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Initializing model on device %s.", device)
    decoder = ConvDecoder(
        embedding_dim=args.latent_emb_dim,
        vocab_size=tokenizer.vocab_size,
        latent_vocab_size=args.latent_vocab_size,
        shrink_factor=args.shrink_factor,
        layers=args.decoder_layers,
        dropout=args.dropout,
        use_dynamic_convolutions=args.decoder_dynamic_convolutions).to(device)
    #decoder = DeconvDecoder(
    #    embedding_dim=args.latent_emb_dim,
    #    vocab_size=tokenizer.vocab_size,
    #    latent_vocab_size=args.latent_vocab_size,
    #    layers=args.decoder_layers,
    #    stride=2,
    #    dropout=args.dropout).to(device)

    model = AutoEncoder(
        args.encoder_type,
        decoder,
        tokenizer.vocab_size,
        latent_vocab_size=args.latent_vocab_size,
        conv_filters=args.convolutions,
        char_embedding_dim=args.char_emb_dim,
        dim=args.dim,
        shrink_factor=args.shrink_factor,
        highway_layers=args.highway_layers,
        char_ff_layers=args.char_ff_layers,
        ff_dim=2 * args.dim,
        layers=args.layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        temperature=args.temperature).to(device)

    loss_function = nn.CTCLoss(zero_infinity=True)
    optimizer = optim.Adam(model.parameters())
    scheduler = NoamLR(optimizer, args.warmup)

    logging.info("Training starts.")
    steps = 0
    updates = 0
    sentences = 0
    best_cer = 9999
    stalled = 0
    for epoch_n in range(args.epochs):
        logging.info("Epoch %d starts, so far %d steps.", epoch_n + 1, updates)
        for src_data, src_mask in train_batches:
            steps += 1
            sentences += src_data.size(0)

            src_data, src_mask = src_data.to(device), src_mask.to(device)

            logits, output_mask = model(
                src_data, src_mask,
                hard_gumbel=steps > args.soft_gumble_steps)

            loss = loss_function(
                F.log_softmax(logits.transpose(0, 1), dim=-1),
                src_data + 1,
                output_mask.sum(1).int(),
                src_mask.sum(1).int())
            loss.backward()

            if steps % args.delay_update == args.delay_update - 1:
                updates += 1
                if (updates % (args.validation_period // 10) ==
                        (args.validation_period // 10) - 1):
                    logging.info(
                        "Step %d, %d sent., temperature %.4g, loss %.4g",
                         updates, sentences, model.temperature, loss.item())
                    tb_writer.add_scalar(
                        "train_loss", loss.item(), global_step=updates)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                torch.cuda.empty_cache()
                if (steps > args.soft_gumble_steps and
                        model.temperature > args.min_temperature):
                    model.temperature *= args.temperature_decrease_factor

                if (updates % args.validation_period ==
                        args.validation_period - 1):
                    val_acc, val_cer, val_bleu, val_loss = validate(
                        model, val_batches, loss_function, device, tokenizer, inference=False,
                        hard_gumbel=steps > args.soft_gumble_steps)

                    logging.info(
                        "TRAIN-like VALIDATION: Step %d (epoch %d), "
                        "Loss: %.4g "
                        "Accuracy: %.4g CER: %.4g  BLEU: %.4g",
                        updates, epoch_n + 1, val_loss.item(),
                        val_acc, val_cer, val_bleu)

                    val_acc, val_cer, val_bleu, val_loss = validate(
                        model, val_batches, loss_function, device, tokenizer)

                    logging.info(
                        "REAL VALIDATION: Step %d (epoch %d), "
                        "Loss: %.4g "
                        "Accuracy: %.4g CER: %.4g  BLEU: %.4g",
                        updates, epoch_n + 1, val_loss.item(),
                        val_acc, val_cer, val_bleu)
                    tb_writer.add_scalar(
                        "val_loss", val_loss.item(), global_step=updates)
                    tb_writer.add_scalar(
                        "val_accuracy", val_acc, global_step=updates)
                    tb_writer.add_scalar(
                        "val_cer", val_cer, global_step=updates)
                    tb_writer.add_scalar(
                        "val_bleu", val_bleu, global_step=updates)
                    if val_cer < best_cer:
                        best_cer = val_cer
                        logging.info("New best CER.")
                        stalled = 0
                    else:
                        stalled += 1
                        logging.info("Best CER so far %.3g, stalled %d times",
                                     best_cer, stalled)

            if stalled >= args.patience:
                break

        if stalled >= args.patience:
            break

    logging.info("Best CER %.3g.", best_cer)
    tb_writer.close()
    logging.info("Done.")


if __name__ == "__main__":
    main()
