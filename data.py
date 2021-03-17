from typing import IO, List, Tuple
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange

from char_tokenizer import BaseTokenizer


T = torch.Tensor


def preprocess_data(
        tokenizer: BaseTokenizer,
        train_src: IO[str],
        train_tgt: IO[str],
        batch_size: int,
        sort_by_length: bool) -> List[Tuple[Tuple[T, T], Tuple[T, T]]]:
    logging.info("Loading file '%s'.", train_src.name)
    src_text = [line.strip() for line in train_src]
    logging.info("Loading file '%s'.", train_tgt.name)
    tgt_text = [line.strip() for line in train_tgt]

    binarized = _binarize_data(tokenizer, src_text, tgt_text, sort_by_length)
    batched = _batch_data(binarized, batch_size, tokenizer.pad_token_id)

    return batched


def _binarize_data(
        tokenizer: BaseTokenizer,
        src_text: List[str], tgt_text: List[str],
        sort_by_length: bool) -> List[Tuple[T, T]]:
    logging.info("Binarizing data.")

    total_sentences = 0
    skipped = 0
    binarized: List[Tuple[T, T]] = []

    pbar = trange(len(src_text), unit="sentences")
    for _, src, tgt in zip(pbar, src_text, tgt_text):
        total_sentences += 1
        if len(src) > 300 or len(tgt) > 300:
            skipped += 1
            continue

        binarized.append(( # type: ignore
            tokenizer.batch_encode_plus(
                [src], return_attention_mask=False)[0],
            tokenizer.batch_encode_plus(
                [tgt], return_attention_mask=False)[0]))

    logging.info(
        "Skipped %d sentences from %d in the dataset.",
        skipped, total_sentences)

    if sort_by_length:
        logging.info("Sorting data by length.")
        binarized.sort(key=lambda p: p[0].size(0))

    return binarized


def _batch_data(
        binarized: List[Tuple[T, T]], max_size: int,
        pad_idx: int) -> List[Tuple[Tuple[T, T], Tuple[T, T]]]:
    batches = []
    max_cur_len = 0

    src_batch: List[T] = []
    tgt_batch: List[T] = []

    logging.info("Batching data.")
    def process_batch() -> None:
        src_pt = pad_sequence(
            src_batch, batch_first=True, padding_value=pad_idx)
        src_mask = (src_pt != pad_idx).float()
        tgt_pt = pad_sequence(
            tgt_batch, batch_first=True, padding_value=pad_idx)
        tgt_mask = (tgt_pt != pad_idx).float()
        batches.append(((src_pt, src_mask), (tgt_pt, tgt_mask)))

    pbar = trange(len(binarized), unit="sentences")
    for _, (src, tgt) in zip(pbar, binarized):
        if src.size(0) > max_size:
            raise ValueError(
                "Single instance is longer than maximum batch size.")
        if len(src_batch) * max(max_cur_len, src.size(0)) > max_size:
            process_batch()
            src_batch, tgt_batch = [], []
            max_cur_len = 0
        src_batch.append(src)
        tgt_batch.append(tgt)
        max_cur_len = max(max_cur_len, src.size(0))

    process_batch()
    return batches
