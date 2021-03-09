"""Tokenizer+vocabulary class for character-level processing.

Implements basics of the Huggingface's tokenizer API.
"""

import typing
from typing import List, Union
from collections import Counter

import numpy as np
import torch
from tqdm import trange

from char_tokenizer import BaseTokenizer, SPECIAL_SYMBOLS, postprocess_idx_list


class BigramTokenizer(BaseTokenizer):
    def __init__(self, tokens: List[str]) -> None:
        super().__init__(tokens)

    def batch_encode_plus(
            self,
            text: Union[str, List[str]],  # the sentence to be encoded
            add_special_tokens: bool = True,  # Add [CLS] and [SEP]
            max_length: int = 512,  # maximum length of a sentence
            truncation: bool = False,
            pad_to_max_length: bool =True,  # Add [PAD]s
            return_attention_mask: bool = True,  # Generate the attention mask
            return_tensors: str = "pt"):

        if not add_special_tokens:
            raise ValueError(
                "Bigram tokenizer does not work without special symbols.")

        if isinstance(text, str):
            text = [text]

        idx_list = []
        for sent in text:
            char_list = list(sent)
            if add_special_tokens:
                char_list = ["<s>"] + char_list + ["</s>"]

            token_list = [self.bos_token_id]
            for i in range(len(char_list) - 1):
                bigram = char_list[i] + char_list[i + 1]
                if bigram in self.str_to_idx:
                    token_list.append(self.str_to_idx[bigram])
                else:
                    token_list.append(
                        self.str_to_idx.get(
                            char_list[i + 1], self.unk_token_id))

            if max_length is not None and len(token_list) > max_length:
                if truncation:
                    token_list = token_list[:max_length]
                else:
                    raise ValueError(
                        "The sequence is too long and trunkation is disabled.")

            idx_list.append(token_list)

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

        chars = []
        for char_id in token_ids:
            if char_id == self.bos_token_id:
                continue
            if char_id in [self.eos_token_id, self.pad_token_id]:
                break

            str_form = self.idx_to_str[char_id]
            if str_form.endswith("</s>"):
                break

            chars.append(str_form[-1])
        return "".join(chars)


def from_data(
        text: List[str],
        max_vocab: int = None,
        max_lines: int = None) -> BigramTokenizer:
    """Create char-level tokenizer from data."""
    unigram_counter: typing.Counter[str] = Counter()
    bigram_counter: typing.Counter[str] = Counter()

    len_limit = len(text)
    if max_lines is not None:
        len_limit = min(max_lines, len_limit)
    pbar = trange(len_limit, unit="sentences")
    for _, sent in zip(pbar, text):
        if not sent:
            continue
        unigram_counter.update(sent)
        bigram_counter.update([f"<s>{sent[0]}", f"{sent[-1]}</s>"])
        bigram_counter.update([
            sent[j] + sent[j + 1] for j in range(len(sent) -1)])
    pbar.close()

    if max_vocab is None:
        vocab_list = list(unigram_counter.keys()) + list(bigram_counter.keys())
    else:
        vocab_list = [
            tok for tok, _ in unigram_counter.most_common(max_vocab)]
        vocab_list += [
            tok for tok, _ in bigram_counter.most_common(max_vocab ** 2)]

    vocab = SPECIAL_SYMBOLS + vocab_list
    return BigramTokenizer(vocab)
