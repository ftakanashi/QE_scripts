#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
This script is modified based no the blueprint of
huggingface/transformers-3.3.1: run_language_modeling.py

Example of usage:
python run_tlm.py --model_name_or_path xlm-roberta --model_type xlm-roberta
--train_src_data_file IR-ALT.ende/test/test.src --train_tgt_data_file IR-ALT.ende/test/test.pe
--mlm --mlm_probability 0.4 --output_dir test_pt_output/ --overwrite_output_dir
--do_train --num_train_epochs 5.0 --learning_rate 3e-5 --logging_step 5
'''


import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


'''
    Here is some customized classes for TLM-pretrain
'''
import torch
from torch.utils.data import Dataset
class ParallelTextDataset(Dataset):

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            src_file_path: str,
            tgt_file_path: str,
            max_seq_length: int,
    ):
        assert os.path.isfile(src_file_path) and os.path.isfile(tgt_file_path), \
            f"Input file path {src_file_path} or {tgt_file_path} not found"
        # block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        max_seq_length -= 2    # exclude first <s> and last </s> in the processed input sequence

        logger.info(f"Creating features from dataset file at {src_file_path} and {tgt_file_path}")

        self.examples = []
        with open(src_file_path, encoding="utf-8") as f:
            src_text_lines = [l.strip() for l in f]
        with open(tgt_file_path, encoding='utf-8') as f:
            tgt_text_lines = [l.strip() for l in f]

        for src_line, tgt_line in zip(src_text_lines, tgt_text_lines):
            full_line = src_line + f" {tokenizer.sep_token} {tokenizer.sep_token} " + tgt_line
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(full_line))
            if len(tokenized_text) > max_seq_length:    # ignore all samples exceeds max_seq_length
                continue
            else:
                self.examples.append(
                    tokenizer.build_inputs_with_special_tokens(tokenized_text)
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


'''
    End customized things
'''

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

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
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # train_data_file: Optional[str] = field(
    #     default=None, metadata={"help": "The input training data file (a text file)."}
    # )
    train_src_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training source data file."}
    )

    train_tgt_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training target data file."}
    )

    # eval_data_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )

    eval_src_data_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation source data file to evaluate the perplexity on"}
    )

    eval_tgt_data_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation target data file to evaluate the perplexity on"}
    )

    # line_by_line: bool = field(
    #     default=False,
    #     metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    # )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        # default=1 / 6,
        default=0.0,
        metadata={
            "help": "Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling."
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    # block_size: int = field(
    #     default=-1,
    #     metadata={
    #         "help": "Optional input sequence length after tokenization."
    #                 "The training dataset will be truncated in block of this size for training."
    #                 "Default to the model max input length for single sentence inputs (take into account special tokens)."
    #     },
    # )

    max_seq_length: int = field(
        default=384, metadata={"help": "Maximum length of input sequence including two </s> between source and target sentences."}
    )

def get_dataset(
        args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
):
    if evaluate:
        src_file_path = args.eval_src_data_file
        tgt_file_path = args.eval_tgt_data_file
    else:
        src_file_path = args.train_src_data_file
        tgt_file_path = args.train_tgt_data_file
    return ParallelTextDataset(
        tokenizer=tokenizer,
        src_file_path=src_file_path,
        tgt_file_path=tgt_file_path,
        max_seq_length=args.max_seq_length
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_src_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_src_data_file "
            "or remove the --do_eval argument."
        )
    if data_args.eval_tgt_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_tgt_data_file "
            "or remove the --do_eval argument."
        )

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

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

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
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

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

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )

    # if data_args.block_size <= 0:
    #     data_args.block_size = tokenizer.max_len
    #     # Our input block size will be the max possible for the model
    # else:
    #     data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets

    train_dataset = (
        get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=tokenizer, evaluate=True)
        if training_args.do_eval
        else None
    )
    if config.model_type == "xlnet":
        data_collator = DataCollatorForPermutationLanguageModeling(
            tokenizer=tokenizer,
            plm_probability=data_args.plm_probability,
            max_span_length=data_args.max_span_length,
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
