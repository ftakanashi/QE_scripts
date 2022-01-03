#!/usr/bin/env python

import argparse
import json
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src",
                        help="Path to the source corpus.")
    parser.add_argument("-t", "--tgt",
                        help="Path to the target corpus.")
    parser.add_argument("-o", "--output",
                        help="Output file.")
    parser.add_argument("--blank_prob", type=float,
                        help="Probability of a token to be blanked.")

    parser.add_argument("--source_lang", default="en_XX",
                        help="Source language. Default: en_XX")
    parser.add_argument("--target_lang", default="zh_CN",
                        help="Target language. Default: zh_CN")
    parser.add_argument("--blank_token", type=str, default="¶",
                        help="Representation of [blank]. Default: '¶'")
    parser.add_argument("--answer_token", type=str, default="※",
                        help="Representation of [answer]. Default: '※'")

    args = parser.parse_args()

    assert 0.0 < args.blank_prob < 1.0, f"Invalid blank_prob {args.blank_prob}"

    return args

def main():
    args = parse_args()

    def read_fn(fn):
        with open(fn) as f:
            return [l.strip() for l in f]

    src_lines = read_fn(args.src)
    tgt_lines = read_fn(args.tgt)
    src_lang = args.source_lang.split("_")[0]
    tgt_lang = args.target_lang.split("_")[0]
    all_line_res = []
    for src_line, tgt_line in zip(src_lines, tgt_lines):

        tgt_tokens = tgt_line.strip().split()

        blanked_tokens = []
        answer_tokens = []
        for i in range(len(tgt_tokens)):
            if random.random() < args.blank_prob:    # do blanking
                if not blanked_tokens or blanked_tokens[-1] != args.blank_token:
                    blanked_tokens.append(args.blank_token)
                else:    # continuous blank
                    answer_tokens.pop()
                answer_tokens.append(tgt_tokens[i])
                answer_tokens.append(args.answer_token)
            else:
                blanked_tokens.append(tgt_tokens[i])

        if len(answer_tokens) == 0:
            continue

        line_res = {
            src_lang: src_line,
            f"{tgt_lang}_blank": " ".join(blanked_tokens),
            tgt_lang: " ".join(answer_tokens)
        }
        all_line_res.append(line_res)

    wf = open(args.output, "w")
    for line_res in all_line_res:
        wf.write(json.dumps({"translation": line_res}, ensure_ascii=False) + "\n")
    wf.close()
    print(f"{len(all_line_res)} pairs of data generated to {args.output}.")

if __name__ == '__main__':
    main()