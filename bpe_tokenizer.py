
import re
from typing import List, Union

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import torch

from char_tokenizer import BaseTokenizer, SPECIAL_SYMBOLS, postprocess_idx_list


class BPETokenizer(BaseTokenizer):
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(SPECIAL_SYMBOLS)
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

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
        for sent in self.tokenizer.encode_batch(text):
            ids = sent.ids
            if add_special_tokens:
                ids = [self.bos_token_id] + ids + [self.eos_token_id]

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

        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        decoded = re.sub("^<s> ", "", decoded)
        decoded = re.sub(" </s>.*", "", decoded)
        decoded = re.sub(" ", "", decoded)
        decoded = re.sub("▁", " ", decoded)

        if decoded.startswith(" "):
            return decoded[1:]
        return decoded


def from_data(
        text: List[str],
        max_vocab: int = None,
        max_lines: int = None) -> Tokenizer:
    tokenizer = Tokenizer(BPE(
        unk_token="<unk>",
        continuing_subword_prefix="▁"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        special_tokens=SPECIAL_SYMBOLS,
        continuing_subword_prefix="▁",
        vocab_size=max_vocab)

    tokenizer.train_from_iterator(text[:max_lines], trainer)

    return BPETokenizer(tokenizer)
