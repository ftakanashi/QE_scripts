# coding: utf-8

import logging
import json
import os

from tqdm import tqdm
from typing import Dict

from torch.utils.data import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import logging

logger = logging.get_logger(__name__)

class AlreadyMaskedLineDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int,
                 data_type="train", with_src=False, **kwargs):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        logger.info("Creating features from dataset file at %s", file_path)
        self.with_src = with_src
        self.tokenizer = tokenizer

        self.is_train = data_type == "train"

        self.src_lang = kwargs.get("src_lang", "en")
        self.tgt_lang = kwargs.get("tgt_lang", "zh")
        self.blank_token = kwargs.get("blank_token", "¶")
        self.answer_token = kwargs.get("answer_token", "※")
        self.mask_n_repeat = kwargs.get("mask_n_repeat", 1)

        logger.info("Start reading raw data...")
        with open(file_path, encoding="utf-8") as f:
            lines = [json.loads(l.strip())["translation"] for l in f]
        logger.info("Raw data finished reading...")

        self.examples = []

        if self.is_train:
            orig_lines, masked_lines = self._organize_train_data(lines)
            for orig_line, masked_line in tqdm(zip(orig_lines, masked_lines), mininterval=1, desc="Processing data"):
                orig_encoding = tokenizer(orig_line, add_special_tokens=True, truncation=True, max_length=block_size)["input_ids"]
                masked_encoding = tokenizer(masked_line, add_special_tokens=True, truncation=True, max_length=block_size)["input_ids"]
                assert len(orig_encoding) == len(masked_encoding), "token number of restored sentence and masked sentence does not match."
                self.examples.append({
                    "input_ids": masked_encoding,
                    "labels": orig_encoding
                })
        else:
            for info in tqdm(lines, mininterval=1, desc="Processing data"):
                with_blank_tokens = info[f"{self.tgt_lang}_blank"].strip().split()
                for token_i, token in enumerate(with_blank_tokens):
                    if token == self.blank_token:
                        with_blank_tokens[token_i] = " ".join([tokenizer.mask_token] * self.mask_n_repeat)
                masked_line = " ".join(with_blank_tokens)
                if self.with_src:
                    masked_line = info[self.src_lang].strip() + f" {tokenizer.sep_token} {tokenizer.sep_token} " + masked_line
                masked_encoding = tokenizer(masked_line, add_special_tokens=True, truncation=True, max_length=block_size)["input_ids"]
                self.examples.append({
                    "input_ids": masked_encoding
                })

    def _organize_train_data(self, lines):
        src, tgt = self.src_lang, self.tgt_lang
        tokenizer = self.tokenizer
        orig_sents = []
        masked_sents = []
        for info in tqdm(lines, mininterval=1, desc="Reading data"):
            answer_spans = [s.strip() for s in info[tgt].split(self.answer_token) if len(s) > 0]

            # restore the original (target) sentence and generate the token-number-matched masked sentence
            with_blank_tokens = info[f"{tgt}_blank"].strip().split()
            restore_orig_tokens = with_blank_tokens.copy()
            i = j = 0
            while i < len(with_blank_tokens) and j < len(answer_spans):
                while with_blank_tokens[i] != self.blank_token: i += 1
                restore_orig_tokens[i] = answer_spans[j]
                with_blank_tokens[i] = " ".join([tokenizer.mask_token] * len(tokenizer.tokenize(answer_spans[j])))
                i += 1
                j += 1
            orig_sent = " ".join(restore_orig_tokens)
            masked_sent = " ".join(with_blank_tokens)

            if self.with_src:
                src_sent = info[src].strip()
                orig_sent = src_sent + f" {tokenizer.sep_token} {tokenizer.sep_token} " + orig_sent
                masked_sent = src_sent + f" {tokenizer.sep_token} {tokenizer.sep_token} " + masked_sent

            orig_sents.append(orig_sent)
            masked_sents.append(masked_sent)

        return orig_sents, masked_sents

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict:
        return self.examples[i]