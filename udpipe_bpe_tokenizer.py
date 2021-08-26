from typing import List, Tuple, Union
import typing
import logging
import os
import tempfile
from collections import Counter

import numpy as np
import spacy_udpipe
from subword_nmt import apply_bpe, learn_bpe
import torch
from tqdm import trange

from char_tokenizer import BaseTokenizer, SPECIAL_SYMBOLS, postprocess_idx_list


class UDPipeBPETokenizer(BaseTokenizer):
    def __init__(self, src_lng: str, tgt_lng: str, bpe: apply_bpe.BPE, vocab: List[str]) -> None:
        super().__init__(vocab)
        self.bpe = bpe

        self.src_lng = src_lng
        self.tgt_lng = tgt_lng

        self._src_udpipe = None
        self._tgt_udpipe = None
        self.pretokenization = None

    def switch_languages(self) -> None:
        tmp_lng = self.src_lng
        self.src_lng = self.tgt_lng
        self.tgt_lng = tmp_lng
        self._src_udpipe = None
        self._tgt_udpipe = None

    @property
    def src_udpipe(self):
        if (hasattr(self, "pretokenization") and
                self.pretokenization in ["skip", "char"]):
            return self.pretokenization

        if self._src_udpipe is None:
            spacy_udpipe.download(self.src_lng)
            self._src_udpipe = spacy_udpipe.load(self.src_lng)
        return self._src_udpipe

    @property
    def tgt_udpipe(self):
        if hasattr(self, "pretokenization") and self.pretokenization == "skip":
            return "skip"

        if self._tgt_udpipe is None:
            spacy_udpipe.download(self.tgt_lng)
            self._tgt_udpipe = spacy_udpipe.load(self.tgt_lng)
        return self._tgt_udpipe

    def batch_encode_plus(
            self,
            text: Union[str, List[str]],  # the sentence to be encoded
            add_special_tokens: bool = True,  # Add [CLS] and [SEP]
            max_length: int = 512,  # maximum length of a sentence
            truncation: bool = False,
            pad_to_max_length: bool =True,  # Add [PAD]s
            return_attention_mask: bool = True,  # Generate the attention mask
            return_tensors: str = "pt"):

        if isinstance(text, str):
            text = [text]

        idx_list = []
        for sent in pretokenize(self.src_udpipe, text):
            # TODO here should be an option to do char-level
            subwords = self.bpe.process_line(" ".join(sent)).split()
            if add_special_tokens:
                subwords = ["<s>"] + subwords + ["</s>"]
            ids = [
                self.str_to_idx.get(tok, self.unk_token_id)
                for tok in subwords]

            if max_length is not None and len(ids) > max_length:
                if truncation:
                    ids = ids[:max_length]
                else:
                    raise ValueError(
                        "The sequence is too long and trunkation is disabled.")
            idx_list.append(ids)

        return postprocess_idx_list(
            idx_list, pad_to_max_length, return_tensors, return_attention_mask)

    def decode(
            self,
            token_ids: Union[int, List[int], np.ndarray, torch.Tensor]) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, np.ndarray):
            assert len(token_ids.shape) == 1
        if isinstance(token_ids, torch.Tensor):
            assert len(token_ids.shape) == 1
            token_ids = token_ids.cpu().numpy()

        return "".join([
            self.idx_to_str[idx]
            for idx in token_ids if idx > 3]).replace("▁", " ")[1:]


def pretokenize(model, sentences):
    tokenized = []
    if model == "skip":
        for sent in sentences:
            tokenized.append(sent.split())
        return tokenized
    if model == "char":
        for sent in sentences:
            tokenized.append(list("▁" + sent.replace(" ", "▁")))
        return tokenized

    pbar = trange(len(sentences), unit="sentences")
    for _, sent in zip(pbar, sentences):
        try:
            analyzed = model(sent, disable=['parser', 'tagger', 'ner'])
            tokenized_sent = []
            for i, tok in enumerate(analyzed):
                if i == 0 or analyzed[i - 1].whitespace_:
                    tokenized_sent.append("▁" + tok.text)
                else:
                    tokenized_sent.append(tok.text)
            tokenized.append(tokenized_sent)
        except ValueError: # There was something weird in Chinese
            tokenized.append(sent.split())
    pbar.close()
    return tokenized


def from_data(
        text: Tuple[List[str], List[str]],
        max_vocab: int = None,
        max_lines: int = None,
        min_frequency: int = None,
        src_language: str = "en",
        tgt_language: str = "de") -> UDPipeBPETokenizer:
    if min_frequency is None:
        raise ValueError(
            "Minimum token frequency must not be defined.")
    if max_vocab is None:
        raise ValueError(
            "With BPE max vocab must be defined.")

    src_text, tgt_text = text

    if max_lines is not None:
        src_text = src_text[:max_lines // 2]
        tgt_text = tgt_text[:max_lines // 2]

    spacy_udpipe.download(src_language)
    src_udpipe = spacy_udpipe.load(src_language)
    logging.info("Pretokenization of the source language.")
    src_pretokenized = pretokenize(src_udpipe, src_text)
    spacy_udpipe.download(tgt_language)
    logging.info("Pretokenization of the target language.")
    tgt_udpipe = spacy_udpipe.load(tgt_language)
    tgt_pretokenized = pretokenize(tgt_udpipe, tgt_text)

    full_vocab: typing.Counter[str] = Counter()
    for sent in src_pretokenized + tgt_pretokenized:
        full_vocab.update(sent)
    vocab_list = [f'{key} {freq}' for key, freq in full_vocab.items()]

    logging.info("Learn the BPE.")
    tmp_codes_file = tempfile.mkstemp()[1]
    with open(tmp_codes_file, "w") as codes:
        learn_bpe.learn_bpe(
            vocab_list, codes, num_symbols=max_vocab,
            min_frequency=min_frequency, is_dict=True)

    with open(tmp_codes_file) as codes:
        bpe = apply_bpe.BPE(codes, separator="")

    logging.info("Pass over training data to get vocab.")
    bpe_vocab = set()
    for sent in src_pretokenized + tgt_pretokenized:
        bpe_vocab.update(bpe.process_line(" ".join(sent)).split())

    os.remove(tmp_codes_file)
    vocab = SPECIAL_SYMBOLS + list(bpe_vocab)

    return UDPipeBPETokenizer(src_language, tgt_language, bpe, vocab)
