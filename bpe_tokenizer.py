import logging
import os
import tempfile
from typing import List, Union

import sacremoses
import numpy as np
import torch
from tqdm import trange
import youtokentome as yttm

from char_tokenizer import BaseTokenizer, SPECIAL_SYMBOLS, postprocess_idx_list


class BPETokenizer(BaseTokenizer):
    def __init__(self, codes: List[str]) -> None:
        super().__init__(SPECIAL_SYMBOLS)
        self.codes = codes
        self._tokenizer: yttm.BPE = None
        self.pretokenizer = sacremoses.MosesTokenizer()
        self.detokenizer = sacremoses.MosesDetokenizer()

    @property
    def tokenizer(self) -> yttm.BPE:
        if self._tokenizer is None:
            tmp_bpe_file = tempfile.mkstemp()[1]
            with open(tmp_bpe_file, "w") as f_bpe:
                for line in self.codes:
                    print(line, file=f_bpe)
            self._tokenizer = yttm.BPE(tmp_bpe_file)
            os.remove(tmp_bpe_file)
            self.idx_to_str = self._tokenizer.vocab()
            self.str_to_idx = {
                s: i for i, s in enumerate(self.idx_to_str)}
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size()

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
            pretok_sent = " ".join(self.pretokenizer.tokenize(sent))
            ids = self.tokenizer.encode(
                pretok_sent, bos=add_special_tokens, eos=add_special_tokens)

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

        token_id_list = token_ids.tolist()
        if self.eos_token_id in token_id_list:
            token_id_list = token_id_list[
                :token_id_list.index(self.eos_token_id)]
        decoded = self.tokenizer.decode(
            token_id_list, ignore_ids=[0, 1, 2, 3])[0]
        return self.detokenizer.detokenize(decoded.split(" "))


def from_data(
        text: List[str],
        max_vocab: int = None,
        max_lines: int = None,
        min_frequency: int = None,
        src_language: str = None,
        tgt_language: str = None) -> BPETokenizer:
    if min_frequency is not None:
        raise ValueError(
            "With BPE minimum token frequency must not be defined.")

    if max_vocab is None:
        raise ValueError(
            "Max vocab must be defined.")


    if max_lines is not None:
        text = text[:max_lines]

    pretokenizer = sacremoses.MosesTokenizer()
    tmp_input_file = tempfile.mkstemp()[1]
    logging.info("Save data in tmp file %s.", tmp_input_file)
    pbar = trange(len(text), unit="sentences")
    with open(tmp_input_file, "w") as f_inp:
        for _, line in zip(pbar, text):
            print(" ".join(pretokenizer.tokenize(line)), file=f_inp)

    tmp_bpe_file = tempfile.mkstemp()[1]

    logging.info("Training BPE.")
    yttm.BPE.train(
        data=tmp_input_file,
        model=tmp_bpe_file,
        vocab_size=max_vocab,
        coverage=0.9999,
        n_threads=-1,
        pad_id=0, unk_id=3, bos_id=1, eos_id=2)

    with open(tmp_bpe_file) as f_bpe:
        codes = [line.strip() for line in f_bpe]

    logging.info("Remove temporary files.")
    os.remove(tmp_input_file)
    os.remove(tmp_bpe_file)

    return BPETokenizer(codes)
