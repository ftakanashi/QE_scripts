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
        
        --source_prob_threshold FLOAT    [only required in testing for regression]
        --mt_word_prob_threshold FLOAT    [only required in testing for regression]
        --mt_gap_prob_threshold FLOAT    [only required in testing for regression]
        --tag_prob_pooling [mean,max,min]   [set the mode for pooling several token tags during prediction]
        --bad_loss_lambda FLOAT    [only optional in training]
    '''

import datetime
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
    buffer = ''
    for tok_i, pieced_token in enumerate(pieced_tokens):
        if pieced_token == '▁': continue
        if pieced_token[0] == '▁':
            pieced_token = pieced_token[1:]
        res[tok_i] = orig_i
        buffer += pieced_token
        if buffer == origin_tokens[orig_i]:
            buffer = ''
            orig_i += 1
    return res


def map_offset_bert(origin_text, tokenizer):
    orig_tokens = origin_text.split()
    pieced_tokens = []
    for token in tokenizer.tokenize(
            origin_text, never_split=tokenizer.all_special_tokens):
        wp_token = tokenizer.wordpiece_tokenizer.tokenize(token)
        if '[UNK]' in wp_token:  # append the original token rather than UNK to avoid match error
            assert len(wp_token) == 1, f'Token {token} is splited by wordpiece but still contains UNK??'
            pieced_tokens.append(token)
        else:
            pieced_tokens.extend(wp_token)

    mapping = {}
    pieced_i = 0

    for orig_i, orig_token in enumerate(orig_tokens):
        tmp_token = pieced_tokens[pieced_i]

        # normalize orig_token (lowercase, accent-norm. No punc-split-norm)
        if orig_token not in tokenizer.all_special_tokens \
                and orig_token not in tokenizer.basic_tokenizer.never_split:
            # special tokens needs no normalization
            if tokenizer.basic_tokenizer.do_lower_case:
                orig_token = tokenizer.basic_tokenizer._run_strip_accents(orig_token)
                orig_token = orig_token.lower()

        # match!
        while True:
            mapping[pieced_i] = orig_i
            pieced_i += 1
            if tmp_token == orig_token:
                break
            else:
                tmp_token += pieced_tokens[pieced_i].replace('##', '')

            if len(tmp_token) > len(orig_token):  # error raising
                msg = f'Original Text:  {" ".join(orig_tokens)}\n' \
                      f'Pieced Text: {" ".join(pieced_tokens)}\n' \
                      f'Original Token: {orig_token}\n' \
                      f'Pieced Tmp Token: {tmp_token}\n' \
                      f'Mapping: {mapping}'
                raise ValueError('Maybe original tokens and pieced tokens does not match.\n' + msg)

    return mapping


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


@dataclass
class QETagClassificationInputFeature:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    word_tag_labels: Optional[List[int]] = None
    gap_tag_labels: Optional[List[int]] = None


class QETagClassificationProcessor(DataProcessor):
    def __init__(self, args):
        self.source_text = args.source_text
        self.mt_text = args.mt_text
        self.source_tags = args.source_tags
        self.mt_word_tags = args.mt_word_tags
        self.mt_gap_tags = args.mt_gap_tags

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

        i = 0
        examples = []
        for src_line, mt_line, source_tags_line, mt_word_tags_line, mt_gap_tags_line in zip(src_lines, mt_lines,
                                                                                            source_tags_lines,
                                                                                            mt_word_tags_lines,
                                                                                            mt_gap_tags_lines):
            guid = f'{set_type}-{i}'
            examples.append(
                QETagClassificationInputExample(guid=guid, source_text=src_line, mt_text=mt_line,
                                                source_tags=source_tags_line,
                                                mt_word_tags=mt_word_tags_line,
                                                mt_gap_tags=mt_gap_tags_line)
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
        for i, e in enumerate(examples):
            origin_text = f'{tokenizer.cls_token} {e.source_text} {tokenizer.sep_token} {tokenizer.sep_token}' \
                          f' {e.mt_text} {tokenizer.sep_token}'
            pieced_to_origin_mapping = map_offset_roberta(origin_text, tokenizer)

            # get token type ids
            pivot1 = len(e.source_text.strip().split()) + 1
            pivot2 = pivot1 + len(e.mt_text.strip().split()) + 2
            token_type_ids = []
            flag = 0
            for i in pieced_to_origin_mapping:
                token_type_ids.append(flag)
                if i == pivot1: flag = 1
            while len(token_type_ids) < args.max_seq_length:
                token_type_ids.append(0)
            batch_token_type_ids.append(token_type_ids)

            if set_type != 'train':
                batch_word_tag_encoding.append(None)
                batch_gap_tag_encoding.append(None)
            else:
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
            feature = QETagClassificationInputFeature(input_ids=text_inputs['input_ids'],
                                                      attention_mask=text_inputs['attention_mask'],
                                                      token_type_ids=token_type_ids,
                                                      word_tag_labels=word_tag_labels,
                                                      gap_tag_labels=gap_tag_labels)
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
from transformers.modeling_bert import BertModel, BertPreTrainedModel
# from transformers.modeling_roberta import RobertaModel
# from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.trainer import Trainer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from typing import Union, Any, NamedTuple, List, Tuple, Optional


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
        has_labels = any(
            inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels", 'word_tag_labels'])

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)
            source_tag_mask, mt_word_tag_mask, mt_gap_tag_mask = generate_source_and_mt_tag_mask(
                inputs['token_type_ids'])
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


class RobertaModelForQETag(BertModel):

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddingsForQETag(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


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
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
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
        metadata={"help": "A lambda factor justifying loss where tag is BAD. Default: 0.1"}
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

        if num_labels == 2:
            source_tag_res_file = os.path.join(training_args.output_dir, 'pred.source_tags')
            mt_word_tag_res_file = os.path.join(training_args.output_dir, 'pred.mtword_tags')
            mt_gap_tag_res_file = os.path.join(training_args.output_dir, 'pred.gap_tags')
        else:
            source_tag_res_file = os.path.join(training_args.output_dir, 'pred.source_tags.prob')
            mt_word_tag_res_file = os.path.join(training_args.output_dir, 'pred.mtword_tags.prob')
            mt_gap_tag_res_file = os.path.join(training_args.output_dir, 'pred.gap_tags.prob')

        if trainer.is_world_master():

            with Path(source_tag_res_file).open('w') as f:
                for tags in orig_source_tag_preds:
                    f.write(' '.join(id_to_label[t] for t in tags) + '\n')

            with Path(mt_word_tag_res_file).open('w') as f:
                for tags in orig_mt_word_tag_preds:
                    f.write(' '.join(id_to_label[t] for t in tags) + '\n')

            with Path(mt_gap_tag_res_file).open('w') as f:
                for tags in orig_mt_gap_tag_preds:
                    f.write(' '.join(id_to_label[t] for t in tags) + '\n')

            with Path(os.path.join(training_args.output_dir, 'gen_config.json')).open('w') as f:
                info = {
                    'time': datetime.datetime.now().strftime('%Y%m%d %H:%M:%S'),
                    'source_prob_threshold': data_args.source_prob_threshold,
                    'mt_word_prob_threshold': data_args.mt_word_prob_threshold,
                    'mt_gap_prob_threshold': data_args.mt_gap_prob_threshold
                }
                for k, v in info.items():
                    f.write(f'{k}: {v}\n')


if __name__ == "__main__":
    main()
