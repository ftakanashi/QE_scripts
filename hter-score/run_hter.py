#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    This script is for fine-tuning and testing of HTER M-BERT, which is designed for predicting HTER score in QE tasks.
    As input of this script, there are source corpus file, MT corpus file, source QE Tags, MT QE tags as well as HTER
    golden scores(only required during training).

    All QE tags are either BAD or OK for every token in corpus. As the MT QE tags data provided by WMT contains the
    tags for GAP, we remained this feature. Therefore, you need not and also must not remove the GAP tags from MT QE
    tags.

    --do_train and --do_eval are two mode options. Please add one for running once.

    An example of using the script for training:
    python run_hter.py --model_name_or_path model --do_train --source_text train.src --mt_text train.mt
    --source_qe_tags train.source_tag --mt_qe_tags train.tag --hter_scores train.hter --max_seq_length 384
    --per_device_train_batch_size 8 --learning_rate 3e-5 --num_train_epochs 3.0 --output_dir fine-tuned
    After training, the fine-tuned model and other configurations will be saved in output_dir.

    An example of using the script for predicting:
    python run_hter.py --model_name_or_path fine-tuned --do_eval --source_text test.src --mt_text test.mt
    --source_qe_tags test.pred.source_tag --mt_te_tags test.pred.tag --max_seq_length 384
    --per_device_eval_batch_size 8 --output_dir predictions
    After predicting, a hter_predictions.txt file will be generated in output_dir
'''

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple, Union, List
import warnings

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
# from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
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
from transformers.data.processors import DataProcessor


@dataclass
class HTERDataArguments:
    source_text: str = field(
        metadata={'help': 'Path to the source text file.'}
    )
    mt_text: str = field(
        metadata={'help': 'Path to the MT text file.'}
    )
    source_qe_tags: str = field(
        metadata={'help': 'Path to the source QE tags file.'}
    )
    mt_qe_tags: str = field(
        metadata={'help': 'Path to the MT QE tags file.'}
    )
    hter_scores: str = field(
        default=None,
        metadata={'help': 'Path to the HTER score file.'}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    do_not_add_gap_token: bool = field(
        default=False,
        metadata={
            'help': 'In default the script will automatically add [GAP] in every gap in MT. If you don\'t want to '
                    'incorporate gap tokens in the input, this option will help you. Notice that you don\'t need to '
                    'change the content of qe_mt_tags because once this option is added, only the word tags namely '
                    'the [1::2] QE tags will be used.'
        }
    )


@dataclass
class HTERScoreInputExample:
    guid: str
    source_text: str
    mt_text: str
    source_qe_tags: str
    mt_qe_tags: str
    hter_score: Optional[str] = None


@dataclass
class HTERScoreInputFeature:
    input_ids: List[int]
    input_tag_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None


class HTERScoreDataProcessor(DataProcessor):
    def __init__(self, args):
        self.source_text = args.source_text
        self.mt_text = args.mt_text
        self.source_qe_tags = args.source_qe_tags
        self.mt_qe_tags = args.mt_qe_tags
        self.hter_scores = args.hter_scores

        self.do_not_add_gap_token = args.do_not_add_gap_token

    def get_examples(self, set_type):

        def read_f(fn):
            with Path(fn).open(encoding='utf-8') as f:
                return [l.strip() for l in f]

        src_lines = read_f(self.source_text)
        mt_lines = read_f(self.mt_text)
        src_qe_tags_lines = read_f(self.source_qe_tags)
        mt_qe_tags_lines = read_f(self.mt_qe_tags)

        assert len(src_lines) == len(mt_lines), 'Inconsistent number of line'
        assert len(src_lines) == len(src_qe_tags_lines), 'Inconsistent number of line'
        assert len(src_lines) == len(mt_qe_tags_lines), 'Inconsistent number of line'

        if set_type in ('train'):
            assert self.hter_scores is not None, 'You need to specify HTER Scores file for training dataset.'
            hter_scores_lines = read_f(self.hter_scores)
        elif set_type == 'eval':
            hter_scores_lines = [None] * len(src_lines)
        else:
            raise ValueError(f'Invalid set_type {set_type}')

        i = 0
        examples = []
        for src_line, mt_line, src_qe_tags_line, mt_qe_tags_line, hter_scores_line in \
                zip(src_lines, mt_lines, src_qe_tags_lines, mt_qe_tags_lines, hter_scores_lines):
            guid = f'{set_type}-{i}'
            if not self.do_not_add_gap_token:
                mt_text = ' '.join(f'[GAP] {t}' for t in mt_line.split()) + ' [GAP]'
            else:
                mt_text = mt_line
            examples.append(
                HTERScoreInputExample(guid=guid, source_text=src_line, mt_text=mt_text, source_qe_tags=src_qe_tags_line,
                                      mt_qe_tags=mt_qe_tags_line, hter_score=hter_scores_line)
            )
            i += 1

        return examples


class HTERScoreDataset(Dataset):

    def __init__(self, args, tokenizer, set_type):
        self.tokenizer = tokenizer
        self.processor = HTERScoreDataProcessor(args)

        msg = f"Creating features from dataset files: {args.source_text}, {args.mt_text}, {args.source_qe_tags}," \
              f" {args.mt_qe_tags}"
        if args.hter_scores is not None: msg += f', {args.hter_scores}'
        logger.info(msg)

        examples = self.processor.get_examples(set_type)

        batch_text_encoding = tokenizer(
            [(e.source_text, e.mt_text) for e in examples],
            max_length=args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        qe_tag_map = {'BAD': 0, 'OK': 1}
        batch_qe_tags_encoding = []
        for i, e in enumerate(examples):
            mt_qe_tags = e.mt_qe_tags.split()
            if args.do_not_add_gap_token:
                mt_qe_tags = mt_qe_tags[1::2]
            # default QE tag for CLS and SEP are OK
            qe_tag_encoding = ['OK'] + e.source_qe_tags.split() + ['OK'] + mt_qe_tags + ['OK']

            origin_text = f'{tokenizer.cls_token} {e.source_text} {tokenizer.sep_token} {e.mt_text} ' \
                          f'{tokenizer.sep_token}'
            # old_pieced_tokens = tokenizer.convert_ids_to_tokens(batch_text_encoding.input_ids[i])
            pieced_to_origin_mapping = self.map_offset(origin_text, tokenizer)
            max_pieced_token_len = max(pieced_to_origin_mapping.keys()) + 1
            pieced_qe_tag_encoding = [qe_tag_encoding[pieced_to_origin_mapping[k]] for k in range(max_pieced_token_len)]
            qe_tag_encoding = pieced_qe_tag_encoding

            while len(qe_tag_encoding) < args.max_seq_length:  # padding adaption
                qe_tag_encoding.append('BAD')  # PAD are set to BAD in default

            if len(qe_tag_encoding) > args.max_seq_length:
                # seems source and mt will be truncated respectively to fit the max_seq_length requirement
                # so it is hard to map offset in that case.
                raise ValueError('I have not done the adaption to qe_tags_input when the text input exceeds max length')

            batch_qe_tags_encoding.append([qe_tag_map[t] for t in qe_tag_encoding])

        hter_scores = [float(e.hter_score) if e.hter_score is not None else None for e in examples]

        self.features = []
        for i in range(len(examples)):
            text_inputs = {k: batch_text_encoding[k][i] for k in batch_text_encoding}
            qe_tags_inputs = batch_qe_tags_encoding[i]
            hter_score = hter_scores[i]
            feature = HTERScoreInputFeature(input_ids=text_inputs['input_ids'],
                                            input_tag_ids=qe_tags_inputs,
                                            attention_mask=text_inputs['attention_mask'],
                                            token_type_ids=text_inputs['token_type_ids'],
                                            label=hter_score)
            self.features.append(feature)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> HTERScoreInputFeature:
        return self.features[i]

    def map_offset(self, origin_text, tokenizer):
        '''
        A very lovely method helps to generate an offset mapping dictionary between original tokens of a sentence
        and the tokens generated by BertTokenizer(or other etc.)
        Made special adaption for punctuations like hyphens.

        todo NOT strictly tested, looks good when working with German.
        '''

        orig_tokens = origin_text.split()

        pieced_tokens = []
        for token in tokenizer.basic_tokenizer.tokenize(origin_text, never_split=tokenizer.all_special_tokens):
            wp_token = tokenizer.wordpiece_tokenizer.tokenize(token)
            if '[UNK]' in wp_token:
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
                orig_token = self.tokenizer.basic_tokenizer._run_strip_accents(orig_token).lower()

            while True:
                mapping[pieced_i] = orig_i
                pieced_i += 1
                if tmp_token == orig_token:
                    break
                else:
                    tmp_token += pieced_tokens[pieced_i].replace('##', '')

                if len(tmp_token) > len(orig_token):
                    msg = f'Original Text:  {" ".join(orig_tokens)}\n' \
                          f'Pieced Text: {" ".join(pieced_tokens)}\n' \
                          f'Original Token: {orig_token}\n' \
                          f'Pieced Tmp Token: {tmp_token}\n' \
                          f'Mapping: {mapping}'
                    raise ValueError('Maybe original tokens and pieced tokens does not match.\n' + msg)

        return mapping


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
import torch
from torch import nn
from torch.nn.modules import MSELoss, CrossEntropyLoss
from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput


class BertModelWithQETag(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        if config.tag_incorporate_mode == 'add':
            self.qe_tag_embeddings = nn.Embedding(2, config.hidden_size)
        elif config.tag_incorporate_mode == 'concat':
            # by default, the dimension of tag embedding is 4.
            self.word_embeddings_extra = nn.Linear(config.hidden_size, config.hidden_size - 4)
            self.qe_tag_embeddings = nn.Embedding(2, 4)
        else:
            raise ValueError(f'Invalid tag_incorporate_mode {config.tag_incorporate_mode}')


        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            input_tag_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
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

        if hasattr(self, 'word_embeddings_extra'):
            # concat
            embedding_output = self.word_embeddings_extra(embedding_output)
            qe_tag_embedding_output = self.qe_tag_embeddings(input_tag_ids)
            embedding_output = torch.cat((embedding_output, qe_tag_embedding_output), dim=-1)
        else:
            # add
            qe_tag_embedding_output = self.qe_tag_embeddings(input_tag_ids)
            embedding_output = embedding_output + qe_tag_embedding_output

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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


class BertModelWithQETagForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModelWithQETag(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.score_outputer = nn.Sigmoid()

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            input_tag_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            input_tag_ids=input_tag_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        score = self.score_outputer(logits)

        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            loss = loss_fct(score.view(-1), labels.view(-1))

        if not return_dict:
            output = (score,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


#############################################################
#
#  End
#
#############################################################

@dataclass
class ModelArguments:
    model_type: str = field(
        metadata={"help": "Type of model"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    tag_incorporate_mode: Optional[str] = field(
        default='add',
        metadata={
            'help': 'The way the model to incorporate QE tag representation.',
            'choices': ['add', 'concat']
        }
    )


def main():

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    parser = HfArgumentParser((ModelArguments, HTERDataArguments, TrainingArguments))

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

    num_labels = 1

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        # finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    config.tag_incorporate_mode = model_args.tag_incorporate_mode

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        never_split=['[GAP]'],
    )
    assert '[GAP]' in tokenizer.vocab, 'You must have the token [GAP] in your vocabulary file in order to let the ' \
                                       'script automatically insert it into MT sentences. To fix this error, ' \
                                       'replace a [unused*] in your vocab.txt with [GAP].'

    model = BertModelWithQETagForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir
    )
    # model = AutoModelForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )

    # Get datasets
    train_dataset = (
        HTERScoreDataset(data_args, tokenizer=tokenizer, set_type='train') if training_args.do_train else None
    )
    eval_dataset = (
        HTERScoreDataset(data_args, tokenizer=tokenizer, set_type='eval') if training_args.do_eval else None
    )

    # def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
    #     def compute_metrics_fn(p: EvalPrediction):
    #         if output_mode == "classification":
    #             preds = np.argmax(p.predictions, axis=1)
    #         elif output_mode == "regression":
    #             preds = np.squeeze(p.predictions)
    #         return glue_compute_metrics(task_name, preds, p.label_ids)
    #
    #     return compute_metrics_fn

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=build_compute_metrics_fn(data_args.task_name),
    )

    # Training
    assert sum(map(int, (training_args.do_train, training_args.do_eval))) == 1, 'You must do train or do eval.'
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logging.info("*** Test ***")

        test_dataset = eval_dataset  # for some reason, I used eval_dataset as the variable's name here...

        predictions = trainer.predict(test_dataset=test_dataset).predictions

        output_test_file = os.path.join(
            training_args.output_dir, f"hter_predictions.txt"
        )
        if trainer.is_world_master():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results HTER Scores *****")
                # writer.write("index\tprediction\n")
                # for index, item in enumerate(predictions):
                #     writer.write("%d\t%3.3f\n" % (index, item))
                for index, item in enumerate(predictions):
                    writer.write(f'{item[0]}\n')


if __name__ == "__main__":
    main()
