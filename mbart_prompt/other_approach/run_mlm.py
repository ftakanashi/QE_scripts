# coding=utf-8

"""
Modified from transformers-v3.3.1 run_language_modeling.py
The script runs MLM only and accept already masked data as input.

python run_mlm.py --model_name_or_path m-bert --model_type bert --do_train --train_data_file train.json
--output_dir output --overwrite_output_dir --overwrite_cache --learning_rate 3e-5 --num_train_epochs 5.0 --logging_steps 10
--do_eval --test_data_file test.json --nbest 5 --mask_n_repeat 1 --results_dir results.m1.n5
"""

import logging
import json
import os
import sys

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
from mbart_prompt.other_approach.myutils.dataset import AlreadyMaskedLineDataset
from mbart_prompt.other_approach.myutils.datacollator import DataCollatorForMaskedLanguageModeling
from mbart_prompt.other_approach.myutils.modeling_xlmr import MyXLMRobertaForMaskedLM
from mbart_prompt.other_approach.myutils.trainer import MyTrainer

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

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
    test_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input testing data file."},
    )
    src_lang: Optional[str] = field(
        default="en_XX",
        metadata={"help": "Source language code."}
    )
    tgt_lang: Optional[str] = field(
        default="zh_CN",
        metadata={"help": "Target language code."}
    )
    blank_token: Optional[str] = field(
        default="¶",
        metadata={"help": "Specify the [blank] token used in data. Default: ¶"}
    )
    answer_token: Optional[str] = field(
        default="※",
        metadata={"help": "Specify the [answer] token used in data. Default: ※"}
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

    nbest: int = field(
        default=5,
        metadata={"help": "output Top N candidates while predicting."}
    )
    mask_n_repeat: int = field(
        default=1,
        metadata={"help": "Number of MASK token inserted for one blank during testing."}
    )
    results_dir: str = field(
        default="results",
        metadata={"help": "Output directory to save the results. It will be generated inside output_dir."}
    )


def get_dataset(
        data_args: DataTrainingArguments,
        model_args: ModelArguments,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool = False,
):
    file_path = data_args.test_data_file if evaluate else data_args.train_data_file
    short_src_lang = data_args.src_lang.split("_")[0]
    short_tgt_lang = data_args.tgt_lang.split("_")[0]
    return AlreadyMaskedLineDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=data_args.block_size,
        data_type="test" if evaluate else "train",
        with_src=model_args.model_type == "xlm-roberta",
        src_lang=short_src_lang,
        tgt_lang=short_tgt_lang,
        blank_token=data_args.blank_token,
        answer_token=data_args.answer_token,
        mask_n_repeat=data_args.mask_n_repeat
    )

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.test_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --test_data_file "
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
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        if model_args.model_type == "xlm-roberta":
            model = MyXLMRobertaForMaskedLM.from_pretrained(
                model_args.model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=model_args.cache_dir
            )
        else:
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
    test_dataset = (
        get_dataset(data_args, model_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    )

    data_collator = DataCollatorForMaskedLanguageModeling(
        tokenizer=tokenizer,
        with_src=model_args.model_type == "xlm-roberta"
    )

    # Initialize our Trainer
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
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
        logger.info("*** Evaluate ***")

        test_dataloader = trainer.get_test_dataloader(test_dataset)
        eval_output = trainer.prediction_loop(test_dataloader, description="Prediction", prediction_loss_only=False,
                                              nbest=data_args.nbest)

        res = analyze_output(eval_output, data_args, model_args, tokenizer)
        result = res["text_res"]
        msg = res["msg_res"]

        results_dir = os.path.join(training_args.output_dir, data_args.results_dir)
        os.makedirs(results_dir, exist_ok=True)
        all_res_output = os.path.join(results_dir, "answer_per_blank.txt")
        rate_output = os.path.join(results_dir, "match_rate.txt")
        if trainer.is_world_master():
            logger.info("***** Eval results *****")
            with open(all_res_output, "w") as writer:
                for row_i, row_res in enumerate(result):
                    writer.write(f"[instance_{row_i}]\n")
                    for mask_cands in row_res:
                        writer.write("\t".join(mask_cands) + "\n")

            wf = open(rate_output, "w")
            wf.write(msg)
            wf.close()

def analyze_output(output, data_args, model_args, tokenizer):
    preds, labels, loss = output

    with open(data_args.test_data_file, "r") as f:
        instances = [json.loads(l.strip())["translation"] for l in f]

    assert len(preds) == len(instances), "Instance number not match between prediction and input test data file."

    # collate continuous mask_n_repeat candidates for one blank
    tgt_short = data_args.tgt_lang.split("_")[0]
    answer_token = data_args.answer_token
    mask_n_repeat = data_args.mask_n_repeat
    res_container = []
    total_cnt = match_1_cnt = match_n_cnt = 0
    for instance_i, instance in enumerate(instances):
        answer_spans = [span for span in instance[tgt_short].strip().split(answer_token) if len(span) > 0]
        instance_preds = preds[instance_i]
        mask_num = instance_preds.shape[0]
        assert mask_num // mask_n_repeat == len(answer_spans), "Answer span number not match"

        instance_res = []
        for span_i, span in enumerate(answer_spans):
            span = "".join(span.strip().split())

            cands = instance_preds[span_i * mask_n_repeat:span_i * mask_n_repeat + mask_n_repeat]
            joined_cands_strs = []
            for blank_cand_ids in cands.T:
                blank_cand_str = "".join(tokenizer.convert_ids_to_tokens(blank_cand_ids))
                if model_args.model_type == "xlm-roberta":
                    tmp = ""
                    for ch in blank_cand_str:
                        if ch == "▁": continue
                        tmp += ch
                    blank_cand_str = tmp
                joined_cands_strs.append(blank_cand_str)

            instance_res.append(joined_cands_strs)
            total_cnt += 1
            if span in joined_cands_strs:
                match_n_cnt += 1
                if span == joined_cands_strs[0]:
                    match_1_cnt += 1

        res_container.append(instance_res)

    # batch_size, seq_len = labels.shape
    # res_container = []
    # total_cnt = match_1_cnt = match_n_cnt = 0
    # for batch_i in range(batch_size):
    #     row_res = []
    #     for token_j in range(seq_len):
    #         label = labels[batch_i, token_j]
    #         if label == -100: continue
    #         total_cnt += 1
    #         cands = pred_scores[batch_i, token_j].topk(data_args.nbest)[1]
    #         if label in cands:
    #             match_n_cnt += 1
    #             if label == cands[0]: match_1_cnt += 1
    #
    #         row_res.append(tokenizer.convert_ids_to_tokens(cands))
    #
    #     res_container.append(row_res)

    msg = f"Top 1 Match: {match_1_cnt / total_cnt:.4f} ({match_1_cnt} / {total_cnt})\n" \
          f"Top n Match: {match_n_cnt / total_cnt:.4f} ({match_n_cnt} / {total_cnt})\n"
    return {"text_res": res_container, "msg_res": msg}

if __name__ == "__main__":
    main()
