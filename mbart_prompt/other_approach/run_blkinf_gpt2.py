#!/usr/bin/env python

# !!! WARN !!!
# Current version do not support running on multi-GPU.

"""
modified from transformers-v3.3.1 run_language_modeling.py

This script takes a multi-line JSON file as input (../build_data.py can generate such format).
It then trains a GPT-2 for CLM with input like below:
Input Sequence: <s> a blanked MT with [BLANK] </s> answer sequence [ANSWER] </s>

Note that this script is ONLY for training GPT-2.
For code for testing and evaluation, refer to ./run_blkinf_gpt2_gen.py

python <this script> --model_name_or_path gpt2-chinese --model_type gpt2 --do_train --train_data_file train.json
--output_dir output --overwrite_output_dir --overwrite_cache --learning_rate 3e-5 --num_train_epochs 5.0 --logging_steps 10
"""

import logging
import os
import sys
import warnings

from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TrainingArguments,
    set_seed,
)

ROOT = os.path.dirname(os.path.abspath(__file__))    # importation adaption
sys.path.append(os.path.dirname(os.path.dirname(ROOT)))
from mbart_prompt.other_approach.myutils.dataset import AlreadyMaskedLineDatasetForCLM
from mbart_prompt.other_approach.myutils.datacollator import DataCollatorForBlankInfilling
from mbart_prompt.other_approach.myutils.trainer import MyTrainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# BLANK_TOKEN = "[BLK]"
# ANSWER_TOKEN = "[ANS]"

@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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


@dataclass
class DataTrainingArguments:

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a already masked file)."}
    )
    # test_data_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "The input testing data file."},
    # )
    src_lang: Optional[str] = field(
        default="en_XX",
        metadata={"help": "Source language code."}
    )
    tgt_lang: Optional[str] = field(
        default="zh_CN",
        metadata={"help": "Target language code."}
    )
    blank_token_in_data: Optional[str] = field(
        default="¶",
        metadata={"help": "Specify the [blank] token used in data. Default: ¶"}
    )
    answer_token_in_data: Optional[str] = field(
        default="※",
        metadata={"help": "Specify the [answer] token used in data. Default: ※"}
    )
    blank_token_for_model: Optional[str] = field(
        default="[BLK]",
        metadata={"help": "Specify the [blank] token for the model. Default: [BLK]"}
    )
    answer_token_for_model: Optional[str] = field(
        default="[ANS]",
        metadata={"help": "Specify the [answer] token for the model. Default: [ANS]"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def get_dataset(
        data_args: DataTrainingArguments,
        model_args: ModelArguments,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
):
    # file_path = data_args.test_data_file if evaluate else data_args.train_data_file
    file_path = data_args.train_data_file
    short_src_lang = data_args.src_lang.split("_")[0]
    short_tgt_lang = data_args.tgt_lang.split("_")[0]
    return AlreadyMaskedLineDatasetForCLM(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=data_args.block_size,
        data_type="test" if evaluate else "train",
        with_src=model_args.model_type == "xlm-roberta",
        src_lang=short_src_lang,
        tgt_lang=short_tgt_lang,
        blank_token_in_data=data_args.blank_token_in_data,
        answer_token_in_data=data_args.answer_token_in_data,
        blank_token_for_model=data_args.blank_token_for_model,
        answer_token_for_model=data_args.answer_token_for_model,
    )

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

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
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    # adapt special tokens for blank infilling so that tokenizer won't split them
    blank_token, answer_token = data_args.blank_token_for_model, data_args.answer_token_for_model

    vocab = tokenizer.get_vocab()
    assert  blank_token in vocab and answer_token in vocab, \
        f"Please manually replace some UNUSED tokens with {blank_token} and {answer_token} or use already included tokens."
    del vocab
    if data_args.tgt_lang == "de_DE":
        tokenizer.add_special_tokens({
            "cls_token": "<s>",
            "sep_token": "</s>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        })

    tokenizer.add_tokens([blank_token, answer_token], special_tokens=True)

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    train_dataset = (
        get_dataset(data_args, model_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    # test_dataset = (
    #     get_dataset(data_args, model_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    # )

    data_collator = DataCollatorForBlankInfilling(tokenizer=tokenizer)

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        prediction_loss_only=True,
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
    if training_args.do_eval:
        raise NotImplementedError("Training is over. But this script is only for training. "
                                  "For testing and evaluation, please refer to run_blkinf_gpt2_gen.py which should be"
                                  " placed in the same position as this script.")

if __name__ == "__main__":
    main()
