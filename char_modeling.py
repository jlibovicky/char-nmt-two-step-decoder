"""Classes handling character-level input and outpu."""

from typing import Dict, Iterable, List, Optional

import torch

T = torch.Tensor


def encode_str(
        sentence: str,
        vocab_dict: Dict[str, int]) -> Optional[torch.LongTensor]:
    if any(c not in vocab_dict for c in sentence):
        return None
    # pylint: disable=not-callable
    return torch.tensor(
        [vocab_dict[c] + 2 for c in list(sentence) + ["</s>"]],
        dtype=torch.int64)
    # pylint: enable=not-callable


def decode_str(logprobs: T, lengths: T, vocab: List[str]) -> Iterable[str]:
    for indices, length in zip(logprobs.argmax(2), lengths):
        word: List[str] = []
        was_blank = True
        for idx in indices[:int(length)]:
            if idx == 1:
                was_blank = True
                continue
            char_to_add = vocab[int(idx) - 2]
            if was_blank or char_to_add != word[-1]:
                word.append(char_to_add)
            was_blank = False
        yield "".join(word)
