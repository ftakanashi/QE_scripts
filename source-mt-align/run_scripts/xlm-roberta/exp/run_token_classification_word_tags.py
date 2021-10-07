# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """

NOTE = \
    '''
        This script is modified from huggingface/transformer's run_ner.py.
        To put it in a simple way, it is a script for token classification, and specifically word-wise word tag prediction.
        Backbone architecture is XLM-RoBERTa.
        
        A typical composition of arguments is like this:
        --model_name_or_path model --do_train --source_text xxx --mt_text xxx --source_tags xxx 
        --mt_word_tags xxx --mt_gap_tags xxx --learning_rate 3e-5 --max_seq_length 384 --output_dir output --cache_dir output 
        --save_steps 1000 --num_train_epochs 5.0 --overwrite_cache --overwrite_output_dir
        
        Some newly added arguments are:
        --source_text FILE
        --mt_text FILE
        --source_tags FILE    [only required in training]
        --mt_word_tags FILE    [only required in training]
        --mt_gap_tags FILE    [only requried in training]
        --valid_tags FILE    [if not set, OK/BAD is the default valid tags.]
        
        --output_prob    [set this flag then you don't need following threshold because probs are output]
        --source_prob_threshold FLOAT    [only required in testing for regression]
        --mt_word_prob_threshold FLOAT    [only required in testing for regression]
        --mt_gap_prob_threshold FLOAT    [only required in testing for regression]
        --tag_prob_pooling [mean,max,min]   [set the mode for pooling several token tags during prediction]
        --bad_loss_lambda FLOAT    [only optional in training]
        
        --alignment_mask
        --source_mt_align FILE
    '''

import datetime
import itertools
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from transformers import (
    AutoConfig,
    # AutoModelForTokenClassification,
    AutoTokenizer,
    # EvalPrediction,
    HfArgumentParser,
    # Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

#############################################################
#
# Start:
# Some of classes and functions about data defined by myself
#
#############################################################

from pathlib import Path
from torch.utils.data import Dataset
from transformers.data import DataProcessor


def map_offset_roberta(origin_text, tokenizer):
    origin_tokens = origin_text.strip().split()
    # never_split = tokenizer.init_kwargs['never_split'] + tokenizer.all_special_tokens
    # pieced_tokens = tokenizer.tokenize(origin_text, never_split=never_split)
    pieced_tokens = tokenizer.tokenize(origin_text)  # todo 需要能够手动调整never_split从而避免分词一些额外特殊token比如<gap>

    res = [-1 for _ in range(len(pieced_tokens))]
    orig_i = 0
    orig_tok_split = tokenizer.tokenize(origin_tokens[0])
    buffer = []
    for tok_i, pieced_token in enumerate(pieced_tokens):
        res[tok_i] = orig_i if pieced_token != '▁' else -1
        buffer.append(pieced_token)
        if buffer == orig_tok_split or ''.join(buffer).replace('▁', '') == origin_tokens[orig_i]:
            orig_i += 1
            if orig_i == len(origin_tokens): break
            orig_tok_split = tokenizer.tokenize(origin_tokens[orig_i])
            buffer = []

    return res


def generate_source_and_mt_tag_mask(token_type_ids):
    '''
    generate masks for source tags and MT tags. Note that CLS and SEP are not included.
    specifically, identifying the min and max indices of 1 in token_type_ids, which respectively identifies the
    first token of MT text and the SEP token of MT text.
    '''
    batch_size, max_len = token_type_ids.shape
    source_tag_masks = []
    mt_tag_masks = []
    mt_gap_tag_masks = []
    for i in range(batch_size):
        row_type_ids = token_type_ids[i, :]
        _mt_indices = row_type_ids.nonzero().view(-1)
        min_i, max_i = _mt_indices.min(), _mt_indices.max()

        source_tag_mask = torch.zeros_like(row_type_ids)
        source_tag_mask[1:min_i - 1] = 1  # CLS and SEP for source text are excluded

        mt_tag_mask = torch.zeros_like(row_type_ids)
        mt_tag_mask[min_i + 1:max_i] = 1  # SEP for MT text is excluded

        mt_gap_tag_mask = torch.zeros_like(row_type_ids)
        mt_gap_tag_mask[min_i + 1:max_i + 1] = 1  # SEP for MT text is included because gaps are one more than MT words

        source_tag_masks.append(source_tag_mask.type(torch.bool))
        mt_tag_masks.append(mt_tag_mask.type(torch.bool))
        mt_gap_tag_masks.append(mt_gap_tag_mask.type(torch.bool))

    source_tag_masks = torch.stack(source_tag_masks, dim=0)
    mt_tag_masks = torch.stack(mt_tag_masks, dim=0)
    mt_gap_tag_masks = torch.stack(mt_gap_tag_masks, dim=0)

    res = (source_tag_masks, mt_tag_masks,
           mt_gap_tag_masks)

    return res


@dataclass
class QETagClassificationInputExample:
    guid: str
    source_text: str
    mt_text: str
    source_tags: Optional[str] = None
    mt_word_tags: Optional[str] = None
    mt_gap_tags: Optional[str] = None
    source_mt_align: Optional[str] = None


@dataclass
class QETagClassificationInputFeature:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    word_tag_labels: Optional[List[int]] = None
    gap_tag_labels: Optional[List[int]] = None
    source_mt_align: Optional[List[Tuple[int]]] = None


class QETagClassificationProcessor(DataProcessor):
    def __init__(self, args):
        self.source_text = args.source_text
        self.mt_text = args.mt_text
        self.source_tags = args.source_tags
        self.mt_word_tags = args.mt_word_tags
        self.mt_gap_tags = args.mt_gap_tags
        self.mt_unk_token_repr = args.mt_unk_token_repr

        if args.alignment_mask:
            assert args.source_mt_align is not None, 'You need to specify source_mt_align if alignment_mask is set.'
            self.source_mt_align = args.source_mt_align

    def get_examples(self, set_type):

        def read_f(fn):
            with Path(fn).open(encoding='utf-8') as f:
                return [l.strip() for l in f]

        src_lines = read_f(self.source_text)
        mt_lines = read_f(self.mt_text)

        assert len(src_lines) == len(mt_lines), 'Inconsistent number of line'

        if set_type == 'train':
            assert self.source_tags is not None, 'You need to specify source QE Tags file to do train.'
            assert self.mt_word_tags is not None, 'You need to specify MT QE Tags file to do train.'
            source_tags_lines = read_f(self.source_tags)
            mt_word_tags_lines = read_f(self.mt_word_tags)
            assert len(src_lines) == len(source_tags_lines), 'Inconsistent number of line'
            assert len(src_lines) == len(mt_word_tags_lines), 'Inconsistent number of line'

            if self.mt_gap_tags is not None:
                mt_gap_tags_lines = read_f(self.mt_gap_tags)
                assert len(src_lines) == len(mt_gap_tags_lines)
            else:
                mt_gap_tags_lines = [None] * len(src_lines)

        elif set_type == 'eval':
            source_tags_lines = mt_word_tags_lines = mt_gap_tags_lines = [None] * len(src_lines)

        else:
            raise ValueError(f'Invalid set type {set_type}')

        if hasattr(self, 'source_mt_align'):
            source_mt_align_lines = read_f(self.source_mt_align)
        else:
            source_mt_align_lines = [None] * len(src_lines)

        i = 0
        examples = []
        for src_line, mt_line, \
            source_tags_line, mt_word_tags_line, mt_gap_tags_line, \
            source_mt_align_line \
                in zip(
            src_lines, mt_lines,
            source_tags_lines, mt_word_tags_lines, mt_gap_tags_lines,
            source_mt_align_lines):

            guid = f'{set_type}-{i}'
            if self.mt_unk_token_repr is not None:
                mt_line = mt_line.replace('[UNK]', self.mt_unk_token_repr)
            examples.append(
                QETagClassificationInputExample(guid=guid, source_text=src_line, mt_text=mt_line,
                                                source_tags=source_tags_line,
                                                mt_word_tags=mt_word_tags_line,
                                                mt_gap_tags=mt_gap_tags_line,
                                                source_mt_align=source_mt_align_line)
            )
            i += 1

        return examples


class QETagClassificationDataset(Dataset):
    def __init__(self, args, tokenizer, set_type, label_to_id):
        self.tokenizer = tokenizer
        self.processor = QETagClassificationProcessor(args)

        if args.source_tags is not None:
            assert args.mt_word_tags is not None, 'You must specify MT QE tags simultaneously'
        else:
            assert args.mt_word_tags is None, 'You must specify Source QE tags simultaneously.'

        if args.mt_gap_tags is not None:
            assert args.mt_word_tags is not None, 'You must specify MT Word tags with MT Gap tags.'

        msg = f"Creating features from dataset files: {args.source_text}, {args.mt_text}"
        if args.source_tags is not None and args.mt_word_tags is not None:
            msg += f', {args.source_tags}, {args.mt_word_tags}'
        if args.mt_gap_tags is not None:
            msg += f', {args.mt_gap_tags}'
        if args.alignment_mask:
            msg += f', {args.source_mt_align}'
        logger.info(msg)

        examples = self.processor.get_examples(set_type)

        batch_text_encoding = tokenizer(
            [(e.source_text, e.mt_text) for e in examples],
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        qe_tag_map = label_to_id
        id_to_label = {i: label for label, i in label_to_id.items()}
        DEF_TAG = id_to_label[0]
        batch_token_type_ids = []
        batch_word_tag_encoding = []
        batch_gap_tag_encoding = []
        batch_source_mt_align = []

        for i, e in enumerate(examples):
            origin_text = f'{tokenizer.cls_token} {e.source_text} {tokenizer.sep_token} {tokenizer.sep_token}' \
                          f' {e.mt_text} {tokenizer.sep_token}'
            pieced_to_origin_mapping = map_offset_roberta(origin_text, tokenizer)
            if len(pieced_to_origin_mapping) > args.max_seq_length:
                continue

            # Read and analyze source-MT alignment information
            if args.alignment_mask:
                original_to_pieced_mapping = collections.defaultdict(list)
                for p, o in enumerate(pieced_to_origin_mapping):
                    original_to_pieced_mapping[o].append(p)

                src_pivot_len = 1
                mt_pivot_len = len(e.source_text.split()) + 2

                raw_alignment = [tuple(map(int, a.split('-'))) for a in e.source_mt_align.split()]
                align_mask_pairs = []
                for src_i, mt_i in raw_alignment:
                    align_mask_pairs.extend(
                        itertools.product(
                            original_to_pieced_mapping[src_i + src_pivot_len],
                            original_to_pieced_mapping[mt_i + mt_pivot_len]
                        )
                    )
                # batch_source_mt_align.append(' '.join([f'{a}-{b}' for a,b in align_mask_pairs]))
                batch_source_mt_align.append(align_mask_pairs)
            else:
                batch_source_mt_align.append(None)

            # get token type ids
            pivot1 = len(e.source_text.strip().split()) + 1
            token_type_ids = []
            flag = 0
            for i in pieced_to_origin_mapping:
                token_type_ids.append(flag)
                if i == pivot1: flag = 1
            while len(token_type_ids) < args.max_seq_length:
                token_type_ids.append(0)
            if 1 not in token_type_ids:
                print(pieced_to_origin_mapping)
                print(origin_text)
                import sys
                sys.exit(1)
            batch_token_type_ids.append(token_type_ids)

            if set_type == 'eval':
                batch_word_tag_encoding.append(None)
                batch_gap_tag_encoding.append(None)
                continue

            source_tags = e.source_tags.split()
            mt_word_tags = e.mt_word_tags.split()
            if e.mt_gap_tags is not None:
                mt_gap_tags = e.mt_gap_tags.split()
                assert len(mt_gap_tags) == len(
                    mt_word_tags) + 1, 'Gap tags should always be one more than the word tags!'
            else:
                mt_gap_tags = None

            # get pieced word tags
            qe_tag_encoding = [DEF_TAG] + source_tags + [DEF_TAG] * 2 + mt_word_tags + [DEF_TAG]
            pieced_qe_tag_encoding = []
            for i in range(len(pieced_to_origin_mapping)):
                pieced_qe_tag_encoding.append(DEF_TAG if pieced_to_origin_mapping[i] < 0 else
                                              qe_tag_encoding[pieced_to_origin_mapping[i]])
            qe_tag_encoding = pieced_qe_tag_encoding
            while len(qe_tag_encoding) < args.max_seq_length:  # padding adaption
                qe_tag_encoding.append(DEF_TAG)  # PADs' tag does not really influence for the mask

            # get pieced gap tags
            qe_gap_tag_encoding = []
            if mt_gap_tags is not None:
                # for convenience we add source tags here but actually they are not effective for the mask
                qe_gap_tag_encoding = [DEF_TAG] + source_tags + [DEF_TAG] * 2 + mt_gap_tags
                pieced_qe_gap_tag_encoding = []
                for i in range(len(pieced_to_origin_mapping)):
                    pieced_qe_gap_tag_encoding.append(DEF_TAG if pieced_to_origin_mapping[i] < 0 else
                                                      qe_gap_tag_encoding[pieced_to_origin_mapping[i]])
                qe_gap_tag_encoding = pieced_qe_gap_tag_encoding
                while len(qe_gap_tag_encoding) < args.max_seq_length:
                    qe_gap_tag_encoding.append(DEF_TAG)

            if max(len(qe_tag_encoding), len(qe_gap_tag_encoding)) > args.max_seq_length:
                # seems source and mt will be truncated respectively to fit the max_seq_length requirement
                # so it is hard to map offset in that case.
                # raise ValueError(
                #     'I have not done the adaption to qe_tags_input when the text input exceeds max length')
                continue

            word_tag_labels = []
            for t in qe_tag_encoding:
                assert t in qe_tag_map, f'{t} is an invalid tag among {",".join(qe_tag_map.keys())}'
                word_tag_labels.append(qe_tag_map[t])

            if len(qe_gap_tag_encoding) > 0:
                gap_tag_labels = []
                for t in qe_gap_tag_encoding:
                    assert t in qe_tag_map, f'{t} is an invalid tag among {",".join(qe_tag_map.keys())}'
                    gap_tag_labels.append(qe_tag_map[t])
            else:
                gap_tag_labels = None

            batch_word_tag_encoding.append(word_tag_labels)
            batch_gap_tag_encoding.append(gap_tag_labels)

        self.features = []
        for i in range(len(batch_word_tag_encoding)):
            text_inputs = {k: batch_text_encoding[k][i] for k in batch_text_encoding}
            token_type_ids = batch_token_type_ids[i]
            word_tag_labels = batch_word_tag_encoding[i]
            gap_tag_labels = batch_gap_tag_encoding[i]
            source_mt_align = batch_source_mt_align[i]
            feature = QETagClassificationInputFeature(input_ids=text_inputs['input_ids'],
                                                      attention_mask=text_inputs['attention_mask'],
                                                      token_type_ids=token_type_ids,
                                                      word_tag_labels=word_tag_labels,
                                                      gap_tag_labels=gap_tag_labels,
                                                      source_mt_align=source_mt_align)
            self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> QETagClassificationInputFeature:
        return self.features[i]


#############################################################
#
#  End
#
#############################################################


#############################################################
#
# Start:
# Some of classes and functions about model defined by myself
#
#############################################################
import collections
import math
import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCELoss
from transformers.data.data_collator import InputDataClass
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler, BertLayer, \
    BertAttention, BertIntermediate, BertOutput, BertSelfAttention, BertSelfOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.trainer import Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from typing import Union, Any, NamedTuple, List, Tuple, Optional


def my_default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.

    20211004 Smartly preprocess source-MT alignment information and add it into features.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif k == "source_mt_align":
                batch[k] = [f[k] for f in features]
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


class QETagClassificationPredictionOutput(NamedTuple):
    source_tag_predictions: np.ndarray
    mt_word_tag_predictions: np.ndarray
    mt_gap_tag_predictions: np.ndarray
    source_tag_mask: torch.Tensor
    mt_word_tag_mask: torch.Tensor
    mt_gap_tag_mask: torch.Tensor
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class QETagClassificationTrainer(Trainer):
    '''
        Since my BertForQETagClassification outputs source and MT logits separately, the trainer must be
        re-implemented to adapt that change since the original trainer could only handle one logit output
        In particular, prediction_step and prediction_loop method needs to be overloaded.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_collator = my_default_data_collator

    def prediction_loop(
            self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> QETagClassificationPredictionOutput:

        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        source_tag_preds, source_tag_masks = [], []
        mt_word_tag_preds, mt_word_tag_masks = [], []
        mt_gap_tag_preds, mt_gap_tag_masks = [], []
        label_ids = []
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm
        samples_count = 0
        for inputs in tqdm(dataloader, desc=description, disable=disable_tqdm):
            batch_loss, \
            batch_source_tag_logits, batch_mt_word_tag_logits, batch_mt_gap_tag_logits, \
            batch_source_tag_masks, batch_mt_word_tag_masks, batch_mt_gap_tag_masks, \
            batch_labels = self.prediction_step(model, inputs, prediction_loss_only)

            batch_size = inputs[list(inputs.keys())[0]].shape[0]
            samples_count += batch_size

            if batch_loss is not None:
                eval_losses.append(batch_loss * batch_size)
            source_tag_preds.append(batch_source_tag_logits)
            source_tag_masks.append(batch_source_tag_masks)
            mt_word_tag_preds.append(batch_mt_word_tag_logits)
            mt_word_tag_masks.append(batch_mt_word_tag_masks)
            mt_gap_tag_preds.append(batch_mt_gap_tag_logits)
            mt_gap_tag_masks.append(batch_mt_gap_tag_masks)
            if batch_labels is not None:
                label_ids.append(batch_labels)

        source_tag_preds = torch.cat(source_tag_preds, dim=0)
        source_tag_masks = torch.cat(source_tag_masks, dim=0)
        mt_word_tag_preds = torch.cat(mt_word_tag_preds, dim=0)
        mt_word_tag_masks = torch.cat(mt_word_tag_masks, dim=0)
        mt_gap_tag_preds = torch.cat(mt_gap_tag_preds, dim=0)
        mt_gap_tag_masks = torch.cat(mt_gap_tag_masks, dim=0)

        if len(label_ids) > 0:
            label_ids = torch.cat(label_ids, dim=0)
        else:
            label_ids = None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if source_tag_preds is not None:
                source_tag_preds = self.distributed_concat(source_tag_preds,
                                                           num_total_examples=self.num_examples(dataloader))
            if mt_word_tag_preds is not None:
                mt_word_tag_preds = self.distributed_concat(mt_word_tag_preds,
                                                            num_total_examples=self.num_examples(dataloader))
            if mt_gap_tag_preds is not None:
                mt_gap_tag_preds = self.distrbuted_concat(mt_gap_tag_preds,
                                                          num_total_examples=self.num_examples(dataloader))
            if label_ids is not None:
                label_ids = self.distributed_concat(label_ids, num_total_examples=self.num_examples(dataloader))

        # Finally, turn the aggregated tensors into numpy arrays.
        if source_tag_preds is not None:
            source_tag_preds = source_tag_preds.cpu().numpy()
        if mt_word_tag_preds is not None:
            mt_word_tag_preds = mt_word_tag_preds.cpu().numpy()
        if mt_gap_tag_preds is not None:
            mt_gap_tag_preds = mt_gap_tag_preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.sum(eval_losses) / samples_count

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return QETagClassificationPredictionOutput(source_tag_predictions=source_tag_preds,
                                                   mt_word_tag_predictions=mt_word_tag_preds,
                                                   mt_gap_tag_predictions=mt_gap_tag_preds,
                                                   source_tag_mask=source_tag_masks,
                                                   mt_word_tag_mask=mt_word_tag_masks,
                                                   mt_gap_tag_mask=mt_gap_tag_masks,
                                                   label_ids=label_ids,
                                                   metrics=metrics)

    def prediction_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], prediction_loss_only: bool
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels", 'word_tag_labels'])

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            source_tag_mask, mt_word_tag_mask, mt_gap_tag_mask = generate_source_and_mt_tag_mask(inputs['token_type_ids'])
            if has_labels:
                loss, logits = outputs[:2]
                source_tag_logits, mt_word_tag_logits, mt_gap_tag_logits = logits
                loss = loss.mean().item()
            else:
                loss = None
                logits = outputs[0]
                source_tag_logits, mt_word_tag_logits, mt_gap_tag_logits = logits
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index if has_labels else self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None, None, None, None, None, None)

        labels = inputs.get("labels")
        if labels is not None:
            labels = labels.detach()
        return (loss, source_tag_logits, mt_word_tag_logits, mt_gap_tag_logits, source_tag_mask, mt_word_tag_mask,
                mt_gap_tag_mask, labels)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        '''
        copied and modified from Trainer.create_optimizer_and_scheduler
        '''
        if self.optimizer is None:

            no_decay = ["bias", "LayerNorm.weight"]

            if hasattr(self.model, 'source_qe_tag_crf'):
                bert_and_classifier_parameters = list(self.model.bert.named_parameters()) + \
                                                 list(self.model.source_tag_outputs.named_parameters()) + \
                                                 list(self.model.mt_word_tag_outputs.named_parameters())

                crf_parameters = list(self.model.source_qe_tag_crf.named_parameters()) + \
                                 list(self.model.mt_qe_tag_crf.named_parameters())

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in bert_and_classifier_parameters if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.learning_rate
                    },
                    {
                        "params": [p for n, p in bert_and_classifier_parameters if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr": self.args.learning_rate
                    },

                    {
                        "params": [p for n, p in crf_parameters if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.crf_learning_rate
                    },
                    {
                        "params": [p for n, p in crf_parameters if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                        "lr": self.args.crf_learning_rate
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )


class BertEmbeddingsWithoutNorm(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        BertLayerNorm = torch.nn.LayerNorm
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings


class RobertaEmbeddingsForQETag(BertEmbeddingsWithoutNorm):

    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.language_embeddings = nn.Embedding(2, config.hidden_size)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)


        base_embedding = super().forward(
            input_ids, token_type_ids=None, position_ids=position_ids, inputs_embeds=inputs_embeds
        )
        language_embedding = self.language_embeddings(token_type_ids)
        embedding = base_embedding + language_embedding
        self.LayerNorm(embedding)
        self.dropout(embedding)
        return embedding

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)

    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
        return incremental_indices.long() + padding_idx


class BertAlignMaskSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super(BertAlignMaskSelfAttention, self).__init__(config)

    def generate_align_mask(self, token_type_ids, source_mt_align):
        batch_size, max_seq_len = token_type_ids.shape
        align_mask = torch.zeros(batch_size, max_seq_len, max_seq_len, dtype=torch.bool, device=token_type_ids.device)

        def get_spec_token_pos(type_ids):
            cls_pos = 0
            sep_pos1 = -1
            for i, type_id in enumerate(type_ids):
                if type_id == 1:
                    sep_pos1 = i - 1
                    break
            sep_pos2 = sep_pos1 + type_ids.sum().data.item()
            return cls_pos, sep_pos1, sep_pos2

        for i in range(batch_size):
            row_token_type_ids = token_type_ids[i, :]
            cls, sep1, sep2 = get_spec_token_pos(row_token_type_ids)
            row_source_mt_align = source_mt_align[i]
            row_mask = align_mask[i]

            row_mask[cls+1:sep1, sep1+1:sep2+1] = True
            row_mask[sep1+1:sep2+1, cls+1:sep1] = True

            for a, b in row_source_mt_align:
                row_mask[a, b] = False
                row_mask[b, a] = False

        return align_mask

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            token_type_ids=None,
            source_mt_align=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if source_mt_align is not None:
            align_mask = self.generate_align_mask(token_type_ids, source_mt_align)
            attention_scores.masked_fill_(align_mask.unsqueeze(1), -10000.0)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertAlignMaskAttention(BertAttention):
    def __init__(self, config):
        super(BertAlignMaskAttention, self).__init__(config)
        self.self = BertAlignMaskSelfAttention(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            token_type_ids=None,
            source_mt_align=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            token_type_ids,
            source_mt_align,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertAlignMaskLayer(BertLayer):
    def __init__(self, config):
        super(BertAlignMaskLayer, self).__init__(config)
        self.attention = BertAlignMaskAttention(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            token_type_ids=None,
            source_mt_align=None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            token_type_ids=token_type_ids,
            source_mt_align=source_mt_align,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs


class BertAlignMaskEncoder(BertEncoder):
    def __init__(self, config):
        super(BertAlignMaskEncoder, self).__init__(config)
        self.config = config
        self.layer = nn.ModuleList([BertAlignMaskLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            token_type_ids=None,
            source_mt_align=None,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                    token_type_ids,
                    source_mt_align,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


class RobertaModelForQETag(BertModel):

    def __init__(self, config):
        super(RobertaModelForQETag, self).__init__(config)

        self.embeddings = RobertaEmbeddingsForQETag(config)
        self.encoder = BertAlignMaskEncoder(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            source_mt_align=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            source_mt_align=source_mt_align,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class RobertaForQETag(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModelForQETag(config)

        self.num_label = len(config.label2id)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.source_tag_outputs = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.mt_word_tag_outputs = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        self.mt_gap_tag_outputs = nn.Sequential(
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

        self.bad_loss_lambda = config.bad_loss_lambda

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            word_tag_labels=None,
            gap_tag_labels=None,
            source_mt_align=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            source_mt_align=source_mt_align,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        source_tag_masks, mt_word_tag_masks, mt_gap_tag_masks = generate_source_and_mt_tag_mask(token_type_ids)
        source_tag_logits = self.source_tag_outputs(sequence_output)
        mt_word_tag_logits = self.mt_word_tag_outputs(sequence_output)
        mt_gap_tag_logits = self.mt_gap_tag_outputs(sequence_output)

        total_loss = None

        if word_tag_labels is not None:

            def calc_loss(logits, labels, masks):
                bce = nn.BCELoss(reduction='sum')
                active_logits = logits.squeeze(-1).masked_fill(~masks, 0.0).view(-1)
                active_labels = labels.masked_fill(~masks, 0).type(torch.float).view(-1)
                if self.bad_loss_lambda != 1.0:
                    bad_weight = torch.ones_like(active_labels).masked_fill_(active_labels.type(torch.bool),
                                                                             self.bad_loss_lambda)
                    bce = BCELoss(reduction='sum', weight=bad_weight)
                return bce(active_logits, active_labels)

            source_loss = calc_loss(source_tag_logits, word_tag_labels, source_tag_masks)
            mt_word_loss = calc_loss(mt_word_tag_logits, word_tag_labels, mt_word_tag_masks)
            if gap_tag_labels is None:
                mt_gap_loss = None
            else:
                mt_gap_loss = calc_loss(mt_gap_tag_logits, gap_tag_labels, mt_gap_tag_masks)

            source_loss /= source_tag_masks.sum()
            mt_word_loss /= mt_word_tag_masks.sum()
            if mt_gap_loss is not None:
                mt_gap_loss /= mt_gap_tag_masks.sum()

            total_loss = source_loss + mt_word_loss
            if mt_gap_loss is not None:
                total_loss += mt_gap_loss

        output = ((source_tag_logits, mt_word_tag_logits, mt_gap_tag_logits),) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output


#############################################################
#
#  End
#
#############################################################

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    # task_type: Optional[str] = field(
    #     default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    # )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})

    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    '''
    ========================================================================================
      @wyzypa
      20210605 add bad_loss_lambda
    ========================================================================================
    '''
    bad_loss_lambda: float = field(
        default=1.0,
        metadata={"help": "A lambda factor justifying loss where tag is BAD. Default: 1.0"}
    )
    '''
    ========================================================================================
      @wyzypa End.
    ========================================================================================
    '''


@dataclass
class DataTrainingArguments:
    source_text: str = field(
        metadata={'help': 'Path to the source text file.'}
    )
    mt_text: str = field(
        metadata={'help': 'Path to the MT text file.'}
    )
    source_tags: str = field(
        default=None,
        metadata={'help': 'Path to the source QE tags file.'}
    )
    mt_word_tags: str = field(
        default=None,
        metadata={'help': 'Path to the MT QE tags file.'}
    )
    mt_gap_tags: str = field(
        default=None,
        metadata={'help': 'Path to the MT Gap tags file.'}
    )
    mt_unk_token_repr: str = field(
        default=None,
        metadata={'help': 'Replace all unknown tokens represented by [UNK] to the specified value.'}
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_lower_case: bool = field(
        default=False,
        metadata={'help': 'Set this flag if you are using an uncased model.'}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    alignment_mask: bool = field(
        default=False,
        metadata={'help': 'Set this flag to activate and apply masks generated from source-MT alignment'}
    )
    source_mt_align: str = field(
        default=None,
        metadata={'help': 'Only used when alignment_mask is set. Path to the source-MT alignment.'}
    )
    output_prob: bool = field(
        default=False,
        metadata={'help': "Set this flag to output all probabilities rather than OK/BAD tags into the output files."}
    )
    source_prob_threshold: float = field(
        default=0.5,
        metadata={
            "help": "The threshold for predicting source tags in regression mode. Only effective during testing when --tag_regression is specified"}
    )
    mt_word_prob_threshold: float = field(
        default=0.5,
        metadata={
            "help": "The threshold for predicting source tags in regression mode. Only effective during testing when --tag_regression is specified"}
    )
    mt_gap_prob_threshold: float = field(
        default=0.9,
        metadata={
            "help": "The threshold for predicting source tags in regression mode. Only effective during testing when --tag_regression is specified"}
    )
    valid_tags: str = field(
        default=None,
        metadata={'help': 'Path to the valid tags file. If not set, tags are expected to be OK/BADs. The most '
                          'frequently used tag is recommended to be placed at first.'}
    )
    tag_prob_pooling: str = field(
        default='max',
        metadata={
            'help': 'Pooling method to merge several token tags into a word tag.',
            'choices': ['mean', 'max', 'min']
        }
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    if data_args.valid_tags is not None:
        with open(data_args.valid_tags, 'r') as f:
            labels = f.read().strip().split()
    else:
        labels = ['OK', 'BAD']

    id_to_label: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label_to_id: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )

    '''
    =================================================================================
      @wyzypa
      20210605 INIT
    =================================================================================
    '''
    if not hasattr(config, 'bad_loss_lambda'):
        config.bad_loss_lambda = model_args.bad_loss_lambda

    '''
    =================================================================================
      @wyzypa End.
    =================================================================================
    '''

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        do_lower_case=data_args.do_lower_case,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
        never_split=['[GAP]']
    )
    model = RobertaForQETag.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        QETagClassificationDataset(args=data_args, tokenizer=tokenizer, set_type='train', label_to_id=label_to_id)
        if training_args.do_train
        else None
    )
    eval_dataset = (
        QETagClassificationDataset(args=data_args, tokenizer=tokenizer, set_type='eval', label_to_id=label_to_id)
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, mask: torch.Tensor) -> List[List[str]]:
        preds = predictions.squeeze(-1)

        batch_size, max_len = preds.shape[:2]
        res = [[] for _ in range(batch_size)]

        mask = mask.cpu().numpy()
        preds[~mask] = -100

        for i in range(batch_size):
            for j in range(1, max_len):
                if preds[i, j] >= 0:
                    res[i].append(preds[i, j])
                elif preds[i, j - 1] >= 0:
                    break

        return res

    def map_tag_to_origin(line_i, text, tokenizer, tags, pred):

        assert pred in ('source', 'mt_word', 'mt_gap'), f'Invalid predicting flag {pred}.'
        if pred == 'source':
            threshold = data_args.source_prob_threshold
        elif pred == 'mt_word':
            threshold = data_args.mt_word_prob_threshold
        else:
            threshold = data_args.mt_gap_prob_threshold

        pieced_to_origin_map = map_offset_roberta(text, tokenizer)
        if pred == 'mt_gap':
            assert len(pieced_to_origin_map) == len(tags) - 1, f'Inconsistent num of tokens in case:\n{text}\n{tags}'
            pieced_to_origin_map.append(max(pieced_to_origin_map) + 1)
        else:
            assert len(pieced_to_origin_map) == len(tags), f'Inconsistent num of tokens in case:\n{text}\n{tags}'

        new_tags = collections.defaultdict(list)
        for i, tag in enumerate(tags):
            if pieced_to_origin_map[i] < 0: continue
            new_tags[pieced_to_origin_map[i]].append(tag)

        res = []
        for i in sorted(new_tags):
            vs = new_tags[i]

            if data_args.tag_prob_pooling == 'mean':
                prob = sum(vs) / len(vs)
            elif data_args.tag_prob_pooling == 'max':
                prob = max(vs)
            elif data_args.tag_prob_pooling == 'min':
                prob = min(vs)
            else:
                prob = max(vs)    # dummy

            if data_args.output_prob:
                res_tag = prob
            else:
                res_tag = 1 if prob >= threshold else 0
            res.append(res_tag)

        if pred == 'mt_gap':
            assert len(res) == len(text.split()) + 1
        else:
            assert len(res) == len(text.split())
        return res

    # Initialize our Trainer
    trainer = QETagClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Predict
    if training_args.do_eval:
        test_dataset = eval_dataset

        source_tag_predictions, mt_word_tag_predictions, mt_gap_tag_predictions, \
        source_tag_mask, mt_word_tag_mask, mt_gap_tag_mask, \
        label_ids, metrics = trainer.predict(test_dataset)

        source_tag_preds = align_predictions(source_tag_predictions, source_tag_mask)
        mt_word_tag_preds = align_predictions(mt_word_tag_predictions, mt_word_tag_mask)
        mt_gap_tag_preds = align_predictions(mt_gap_tag_predictions, mt_gap_tag_mask)

        orig_source_tag_preds = []
        with Path(data_args.source_text).open(encoding='utf-8') as f:
            src_lines = [l.strip() for l in f]
            line_i = 1
            for src_line, source_tag_pred in zip(src_lines, source_tag_preds):
                orig_source_tag_preds.append(
                    map_tag_to_origin(line_i, src_line, tokenizer, source_tag_pred, pred='source'))
                line_i += 1

        orig_mt_word_tag_preds = []
        orig_mt_gap_tag_preds = []
        with Path(data_args.mt_text).open(encoding='utf-8') as f:
            mt_lines = [l.strip() for l in f]
            line_i = 1
            for mt_line, mt_word_tag_pred, mt_gap_tag_pred in \
                    zip(mt_lines, mt_word_tag_preds, mt_gap_tag_preds):
                orig_mt_word_tag_preds.append(
                    map_tag_to_origin(line_i, mt_line, tokenizer, mt_word_tag_pred, pred='mt_word'))
                orig_mt_gap_tag_preds.append(
                    map_tag_to_origin(line_i, mt_line, tokenizer, mt_gap_tag_pred, pred='mt_gap'))
                line_i += 1

        if data_args.output_prob:
            source_tag_res_file = os.path.join(training_args.output_dir, 'pred.source_tags.prob')
            mt_word_tag_res_file = os.path.join(training_args.output_dir, 'pred.mtword_tags.prob')
            mt_gap_tag_res_file = os.path.join(training_args.output_dir, 'pred.gap_tags.prob')
        else:
            source_tag_res_file = os.path.join(training_args.output_dir, 'pred.source_tags')
            mt_word_tag_res_file = os.path.join(training_args.output_dir, 'pred.mtword_tags')
            mt_gap_tag_res_file = os.path.join(training_args.output_dir, 'pred.gap_tags')


        if trainer.is_world_master():

            with Path(source_tag_res_file).open('w') as f:
                for tags in orig_source_tag_preds:
                    if data_args.output_prob:
                        f.write(' '.join([str(p) for p in tags]) + "\n")
                    else:
                        f.write(' '.join(id_to_label[t] for t in tags) + '\n')

            with Path(mt_word_tag_res_file).open('w') as f:
                for tags in orig_mt_word_tag_preds:
                    if data_args.output_prob:
                        f.write(' '.join([str(p) for p in tags]) + "\n")
                    else:
                        f.write(' '.join(id_to_label[t] for t in tags) + '\n')

            with Path(mt_gap_tag_res_file).open('w') as f:
                for tags in orig_mt_gap_tag_preds:
                    if data_args.output_prob:
                        f.write(' '.join([str(p) for p in tags]) + "\n")
                    else:
                        f.write(' '.join(id_to_label[t] for t in tags) + '\n')

            with Path(os.path.join(training_args.output_dir, 'gen_config.json')).open('w') as f:
                info = {
                    'time': datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'),
                }
                if data_args.output_prob:
                    info['output_prob'] = True
                else:
                    extra = {
                        'source_prob_threshold': data_args.source_prob_threshold,
                        'mt_word_prob_threshold': data_args.mt_word_prob_threshold,
                        'mt_gap_prob_threshold': data_args.mt_gap_prob_threshold
                    }
                    info.update(extra)
                for k, v in info.items():
                    f.write(f'{k}: {v}\n')


if __name__ == "__main__":
    main()
