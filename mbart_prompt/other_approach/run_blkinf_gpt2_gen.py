#!/usr/bin/env python

# !!! WARN !!!
# Current version do not support running on multi-GPU.

"""
modified from transformers-v3.3.1 run_generation.py

This script takes a multi-line JSON file as input (../build_data.py can generate such format).
After auto-regressively reading in an input sequence like below:
Input Sequence: <s> a blanked MT with [BLANK] </s>
the trained model of GPT-2 is asked to auto-regressively generate an output sequence like below:
Output Sequence: answer sequence [ANSWER] </s>

Then the script will calculate the top-1 match rate and top-n match rate for each blank.
The results are output to --output_dir.

Example of usage:
python <this script> --model_name_or_path gpt2-chinese --model_type gpt2
--test_data_file test.json --output_dir output --length 64 --stop_token [SEP]
--num_return_sequences 10
"""


import argparse
import json
import logging
import os

import numpy as np
import torch

from tqdm import tqdm

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    AutoTokenizer,
)

# BLANK_TOKEN = "[BLK]"
# ANSWER_TOKEN = "[ANS]"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length

def str_to_span(s, delimeter):
    ans = []
    for span in s.strip().split(delimeter):
        if span == "": continue
        ans.append("".join([ch for ch in span if ch != " "]))
    return ans

def analyze_generated_sequences(generated_sequences, labels, args):
    cand_num = len(generated_sequences)
    blank_num = len(labels)

    answer_per_blank = [[None for _ in range(cand_num)] for _ in range(blank_num)]
    for seq_i, seq in enumerate(generated_sequences):
        seq_preds = str_to_span(seq, args.answer_token_for_model)
        for blank_i in range(min(len(seq_preds), blank_num)):
            answer_per_blank[blank_i][seq_i] = seq_preds[blank_i]

    match_1_cnt = match_n_cnt = 0
    for label_i, label in enumerate(labels):
        if label in answer_per_blank[label_i]:
            if label == answer_per_blank[label_i][0]:
                match_1_cnt += 1
            match_n_cnt += 1
    return match_1_cnt, match_n_cnt, answer_per_blank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument("--test_data_file", type=str, required=True, help="Path to the test data file (in form of JSON).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")

    parser.add_argument("--blank_token_in_data", type=str, default="¶", help="The special token expressing a blank in the data.")
    parser.add_argument("--answer_token_in_data", type=str, default="※", help="The special token concatenating answer sequences in the data.")
    parser.add_argument("--blank_token_for_model", type=str, default="[BLK]", help="The special token expressing a blank for the model.")
    parser.add_argument("--answer_token_for_model", type=str, default="[ANS]", help="The special token concatenating answer sequences for the model.")
    parser.add_argument("--src_lang", type=str, default="en_XX", help="Source language code.")
    parser.add_argument("--tgt_lang", type=str, default="zh_CN", help="Target language code.")

    # parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)

    parser.add_argument("--prefix", type=str, default="", help="Text added prior to input.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    # 20220213 add argument
    parser.add_argument("--num_beams", type=int, default=1, help="The number of beams to do beam search.")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    set_seed(args)

    # Initialize the model and tokenizer
    try:
        args.model_type = args.model_type.lower()
        model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    except KeyError:
        raise KeyError("the model {} you specified is not supported. You are welcome to add it and open a PR :)")

    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    blank_token, answer_token = args.blank_token_for_model, args.answer_token_for_model
    vocab = tokenizer.get_vocab()
    assert  blank_token in vocab and answer_token in vocab, \
        f"Please manually replace some UNUSED tokens with {blank_token} and {answer_token} or use already included tokens."
    del vocab
    if args.tgt_lang == "de_DE":
        tokenizer.add_special_tokens({
            "cls_token": "<s>",
            "sep_token": "</s>",
            "pad_token": "<pad>",
            "mask_token": "<mask>",
        })
    tokenizer.add_tokens([blank_token, answer_token], special_tokens=True)

    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)

    if args.fp16:
        model.half()

    args.length = adjust_length_to_model(args.length, max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    with open(args.test_data_file, "r") as f:
        lines = [json.loads(l.strip())["translation"] for l in f]

    batch_generated_sequences = []
    batch_answer_per_blank = []
    batch_labels = []
    short_tgt_lang = args.tgt_lang.split("_")[0]
    total_cnt, total_match_1_cnt, total_match_n_cnt = 0, 0, 0
    for info in tqdm(lines, mininterval=1, desc="Generating"):

        blanked_input = info[f"{short_tgt_lang}_blank"]
        blanked_input_tokens = blanked_input.strip().split()
        blank_cnt = 0
        for i in range(len(blanked_input_tokens)):
            if blanked_input_tokens[i] == args.blank_token_in_data:
                blanked_input_tokens[i] = args.blank_token_for_model    # replace all blank token in data with that for model which is a manually added token in GPT's vocab
                blank_cnt += 1
        blanked_input_sequence = " ".join(blanked_input_tokens)

        input_ids = tokenizer.encode(blanked_input_sequence, add_special_tokens=True, return_tensors="pt")
        input_ids = input_ids.to(args.device)

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        labels = str_to_span(info[short_tgt_lang], args.answer_token_in_data)
        batch_labels.append(labels)
        total_cnt += len(labels)

        generated_sequences = []
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)    # Decode text
            ans_seq_start = text.find(args.stop_token) + len(args.stop_token) + 1    # extra 1 is for the space after stop token
            ans_seq_end = text.find(args.stop_token, ans_seq_start)
            text = text[ans_seq_start:ans_seq_end - 1] if ans_seq_end > 0 else text[ans_seq_start:]
            generated_sequences.append(text)

        match_1_cnt, match_n_cnt, answer_per_blank = analyze_generated_sequences(generated_sequences, labels, args)
        total_match_1_cnt += match_1_cnt
        total_match_n_cnt += match_n_cnt

        batch_generated_sequences.append(generated_sequences)
        batch_answer_per_blank.append(answer_per_blank)

    def write_fn(fn, content):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(content)

    os.makedirs(args.output_dir, exist_ok=True)

    answer_per_seq_str = ""
    for instance_i, generated_sequences in enumerate(batch_generated_sequences):
        answer_per_seq_str += f"[Instance {instance_i}]\n"
        rows = []
        for seq in generated_sequences:
            spans = str_to_span(seq, args.answer_token_for_model)
            blank_num = len(batch_labels[instance_i])
            spans = spans[:blank_num]
            while len(spans) < blank_num:
                spans.append("null")
            rows.append("\t".join(spans))
        answer_per_seq_str += "\n".join(rows)
        answer_per_seq_str += "\n"
    write_fn(os.path.join(args.output_dir, "answer_per_seq.txt"), answer_per_seq_str)

    answer_per_blank_str = ""
    for instance_i, answer_per_blank in enumerate(batch_answer_per_blank):
        answer_per_blank_str += f"[Instance {instance_i}]\n"
        answer_per_blank_str += "\n".join(
            ["\t".join([answer if answer is not None else "null" for answer in answers])
             for answers in answer_per_blank]
        )
        answer_per_blank_str += "\n"
    write_fn(os.path.join(args.output_dir, "answer_per_blank.txt"), answer_per_blank_str)

    match_rate_str = f"Top 1 Match: {total_match_1_cnt / total_cnt:.4f} ({total_match_1_cnt}/{total_cnt})\n" \
                     f"Top n Match: {total_match_n_cnt / total_cnt:.4f} ({total_match_n_cnt}/{total_cnt})\n"
    write_fn(os.path.join(args.output_dir, "match_rate.txt"), match_rate_str)

    return batch_generated_sequences

if __name__ == "__main__":
    main()