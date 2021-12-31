#!/usr/bin/env python

"""
This script is modified from transformers-v4.12.4/examples/translation/run_translation.py
Usage:
python run_mbart_prompt.py --model_name_or_path mbart-large-cc25 --source_lang en_XX --target_lang zh_CN
--do_train --train_file train.json --learning_rate 3e-5 --per_device_train_batch_size 8 --num_train_epochs 5.0
--output_dir output --overwrite_cache --logging_steps 10
--do_eval --test_file test.json --predict_with_generate
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import datetime
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    default_data_collator,
    set_seed,
)

ROOT = os.path.dirname(os.path.abspath(__file__))    # import adaption
sys.path.append(os.path.dirname(ROOT))
from mbart_prompt.myutils.data_collator import DataCollatorForSeq2Seq
from mbart_prompt.myutils.modeling_bart import MBartForConditionalGeneration
from mbart_prompt.myutils.tokenization_mbart import AdaptMBartTokenizer as MBartTokenizer
from mbart_prompt.myutils.trainer import Seq2SeqTrainer
from mbart_prompt.myutils.training_args import Seq2SeqTrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:

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
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    answer_token: Optional[str] = field(
        default="※",
        metadata={"help": "Specify the [answer] token used in data. Default: ※"}
    )
    merge_space_evaluate: bool = field(
        default=True,
        metadata={"help": "Set this flag to merge spaces between tokens when doing evaluation."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.test_file \
                is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension == "json", "`train_file` should be a json file."
        if self.test_file is not None:
            extension = self.test_file.split('.')[-1]
            assert extension == "json", "test_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def analyze_result(data_args, res_container, target_lang):
    generated_lines = res_container["translations"]
    with open(data_args.test_file) as f:
        ref_lines = [l.strip() for l in f]

    assert len(ref_lines) == len(generated_lines), "Line number of test_file and generated content does not match."

    answer_token = data_args.answer_token

    all_answer_info = {
        "answer_matrix": [],
        "answer_matrix_trans": [],
        "top_1_match": [],
        "top_n_match": []
    }
    for line_i in range(len(ref_lines)):
        ref_info = json.loads(ref_lines[line_i])["translation"]
        gen_seqs = generated_lines[f"instance_{line_i}"]

        labels = []
        for span in ref_info[target_lang].strip().split(answer_token):
            if span == "": continue
            if data_args.merge_space_evaluate:
                labels.append("".join([ch for ch in span if ch != " "]))
            else:
                labels.append(span.strip())

        num_beam = len(gen_seqs)
        num_span = len(labels)
        answer_matrix = [[None for _ in range(num_span)] for _ in range(num_beam)]
        for seq_i, seq in enumerate(gen_seqs):
            seq_spans = []
            for span in seq.strip().split(answer_token):
                if span == "": continue
                if data_args.merge_space_evaluate:
                    seq_spans.append("".join([ch for ch in span if ch != " "]))
                else:
                    seq_spans.append(span.strip())

            for span_i, seq_span in enumerate(seq_spans):
                if span_i == len(answer_matrix[0]): break
                answer_matrix[seq_i][span_i] = seq_span

        all_answer_info["answer_matrix"].append(answer_matrix)

        # transpose
        answer_matrix_trans = [[None for _ in range(num_beam)] for _ in range(num_span)]
        for i in range(num_beam):
            for j in range(num_span):
                answer_matrix_trans[j][i] = answer_matrix[i][j]

        all_answer_info["answer_matrix_trans"].append(answer_matrix_trans)

        # do the math
        top_1_match = [False for _ in range(num_span)]
        top_n_match = [False for _ in range(num_span)]
        for span_i in range(num_span):
            if labels[span_i] == answer_matrix_trans[span_i][0]:
                top_1_match[span_i] = True
                top_n_match[span_i] = True
            elif labels[span_i] in answer_matrix_trans[span_i]:
                top_n_match[span_i] = True
        all_answer_info["top_1_match"].append(top_1_match)
        all_answer_info["top_n_match"].append(top_n_match)

    return all_answer_info

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = MBartTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    )
    model = MBartForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    # model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train` and/or `do_eval`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    assert data_args.target_lang is not None and data_args.source_lang is not None, (
        f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
        "--target_lang arguments."
    )

    tokenizer.src_lang = data_args.source_lang
    tokenizer.tgt_lang = data_args.target_lang

    # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
    # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
    forced_bos_token_id = (
        tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
    )
    model.config.forced_bos_token_id = forced_bos_token_id

    # Get the language codes for input/target.
    source_lang = data_args.source_lang.split("_")[0]
    target_lang = data_args.target_lang.split("_")[0]
    target_blank_lang = f"{target_lang}_blank"

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = [ex[source_lang] for ex in examples["translation"]]
        target_blanks = [ex[target_blank_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        inputs = [prefix + inp for inp in inputs]

        # model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # tokenizer.set_tgt_lang_special_tokens(data_args.target_lang)
        # labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        # model_inputs["labels"] = labels["input_ids"]

        batch = tokenizer.prepare_seq2seq_batch(
            src_texts=inputs, src_lang=data_args.source_lang, tgt_blank_texts=target_blanks, tgt_texts=targets,
            tgt_lang=data_args.target_lang, max_length=data_args.max_source_length, max_target_length=max_target_length,
            padding=True, truncation=True     # padding不设置成True就报错…
        )
        batch['input_ids'] = batch['input_ids'].tolist()
        batch['labels'] = batch['labels'].tolist()
        batch['attention_mask'] = batch['attention_mask'].tolist()

        # 去除pad
        for row in batch['input_ids']:
            while row[-1] == tokenizer.pad_token_id:
                row.pop()
        for row in batch['labels']:
            while row[-1] == tokenizer.pad_token_id:
                row.pop()

        return batch

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        # metrics = predict_results.metrics
        # max_predict_samples = (
        #     data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        # )
        # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                res_container = {"translations": {}}
                for i, candidates in enumerate(predictions):
                    res_container["translations"][f"instance_{i}"] = [
                        cand.strip() for cand in candidates
                    ]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write(json.dumps(res_container, indent=4, ensure_ascii=False))

                def write_fn(fn, content):
                    with open(os.path.join(training_args.output_dir, fn), "w", encoding="utf-8") as f:
                        f.write(content)

                analysis = analyze_result(data_args, res_container, target_lang)
                timestp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                generated_results_dir = os.path.join(training_args.output_dir, "generated_results")
                os.makedirs(generated_results_dir, exist_ok=True)

                answer_matrix_str = f"time: {timestp}\n\n"
                for i, instance in enumerate(analysis["answer_matrix"]):
                    answer_matrix_str += f"\n[Instance {i}]\n"
                    answer_matrix_str += "\n".join(
                        ["\t".join([span if span else "null" for span in seq]) for seq in instance]
                    )
                write_fn(os.path.join(generated_results_dir, "answer_per_seq.txt"), answer_matrix_str)

                answer_matrix_transpose_str = f"time: {timestp}\n\n"
                for i, instance in enumerate(analysis["answer_matrix_trans"]):
                    answer_matrix_transpose_str += f"\n[Instance {i}]\n"
                    answer_matrix_transpose_str += "\n".join(
                        ["\t".join([cand if cand else "null" for cand in blank]) for blank in instance]
                    )
                write_fn(os.path.join(generated_results_dir, "answer_per_blank.txt"), answer_matrix_transpose_str)

                top_1_true = top_1_total = 0
                for instance in analysis["top_1_match"]:
                    for flag in instance:
                        if flag: top_1_true += 1
                        top_1_total += 1

                top_n_true = top_n_total = 0
                for instance in analysis["top_n_match"]:
                    for flag in instance:
                        if flag: top_n_true += 1
                        top_n_total += 1
                msg = f"Top 1 Match: {top_1_true / top_1_total:.4f} ({top_1_true}/{top_1_total})\n" \
                      f"Top n Match: {top_n_true / top_1_total:.4f} ({top_n_true}/{top_n_total})"
                write_fn(os.path.join(generated_results_dir, "match_rate.txt"), msg)

    return results


if __name__ == "__main__":
    main()