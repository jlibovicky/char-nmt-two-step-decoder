"""Tokenizer+vocabulary class for character-level processing.

Implements basics of the Huggingface's tokenizer API.
"""

from typing import List, Union

import numpy as np
import torch


SPECIAL_SYMBOLS = ["<pad>", "<s>", "</s>", "<unk>"]


class CharTokenizer(object):
    def __init__(self, tokens: List[str]) -> None:
        super().__init__()
        self.idx_to_str = tokens
        self.str_to_idx = {s: i for i, s in enumerate(tokens)}

        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        assert tokens[:4] == SPECIAL_SYMBOLS

    @property
    def vocab_size(self) -> int:
        return len(self.idx_to_str)

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
        for sent in text:
            char_list = list(sent)
            if add_special_tokens:
                char_list = ["<s>"] + char_list + ["</s>"]
            if max_length is not None and len(char_list) > max_length:
                if truncation:
                    char_list = char_list[:max_length]
                else:
                    raise ValueError(
                        "The sequence is too long and trunkation is disabled.")

            idx = [self.str_to_idx.get(char, 3) for char in char_list]

            idx_list.append(idx)

        if pad_to_max_length:
            max_length = max(len(i) for i in idx_list)
            idx_list = [
                idx + [0] * (max_length - len(idx)) for idx in idx_list]

        if return_tensors == "pt":
            # pylint: disable=not-callable
            if pad_to_max_length:
                idx_list = torch.tensor(idx_list)
            else:
                idx_list = [torch.tensor(idx) for idx in idx_list]
            # pylint: enable=not-callable
        elif return_tensors == "np":
            if pad_to_max_length:
                idx_list = np.array(idx_list)
            else:
                idx_list = [np.array(id) for idx in idx_list]
        else:
            raise ValueError(f"Unsupported tensor type: {return_tensors}.")

        if return_attention_mask:
            if not pad_to_max_length:
                raise ValueError(
                    "Masking does not make sense without padding.")
            mask = idx_list != 0
            if return_tensors == "pt":
                mask = mask.float()
            return idx_list, mask
        return idx_list

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
            chars.append(self.idx_to_str[char_id])
        return "".join(chars)

    def batch_decode(
            self,
            token_ids: Union[List[List[int]], np.ndarray, torch.Tensor]) -> List[str]:
        if isinstance(token_ids, np.ndarray):
            assert len(token_ids.shape) == 2
        if isinstance(token_ids, torch.Tensor):
            assert len(token_ids.shape) == 2

        return [self.decode(sent) for sent in token_ids]

def from_data(text: List[str], max_lines: int = None) -> CharTokenizer:
    vocab_set = set()
    for i, sent in enumerate(text):
        if max_lines is not None and i >= max_lines:
            break
        for char in sent:
            vocab_set.add(char)

    vocab = SPECIAL_SYMBOLS + sorted(vocab_set)
    return CharTokenizer(vocab)
