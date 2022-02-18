#!/usr/bin/env python

# !!! DEPRECATED !!!

"""
For BERT/XLM-R architectures, we don't know in advance how many masks correspond to a blank.
So we simply try out most possibilities, for example, transferring one blank into 1/2/3 masks and do prediction separately.
Accordingly, we get 3 patterns of predictions for each blank.
Currently, we think that if any pattern match the correct answer, it matches.
This script receive 3 patterns (or more) of answer_per_blank prediction files as well as the data file containing reference
as input and then do a comprehensive evaluation considering all those possibilities.
"""

import argparse
import json
import os
import re

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_file",
                        help="Test data file contains reference of all answer spans.")
    parser.add_argument("--preds", nargs="+",
                        help="All prediction directories containing respective answer_per_blank.txt")

    parser.add_argument("--tgt_lang", default="zh_CN",
                        help="Target language code. Default: zh_CN")
    parser.add_argument("--answer_token", default="※",
                        help="Representation of [answer]. Default: '※'")
    parser.add_argument("--output_dir", default=os.path.join(os.getcwd(), "results.jointly"),
                        help="A directory to save the output file.")
    parser.add_argument("--match_standard", default="character", choices=["character", "token"],
                        help="Standard to judge whether generated answer matches label."
                             "Set 'character' if the target language is character-based ones like Chinese or Japanese."
                             "Set 'token' if the target language is token-based ones like German or English.")

    args = parser.parse_args()
    return args

def parse_answer_per_blank(lines):
    res = []
    ptn = re.compile("^\[instance_(\d)+\]$")
    tmp = []
    for line in lines:
        if ptn.match(line.strip()):
            res.append(tmp.copy())
            tmp.clear()
        else:
            tmp.append(line.strip())
    res.append(tmp.copy())
    res.pop(0)
    return res

def parse_data_file(infos, args):
    res = []
    tgt = args.tgt_lang.split("_")[0]
    answer_token = args.answer_token
    for info in infos:
        answer_spans = info[tgt].strip().split(answer_token)
        if args.match_standard == "character":
            answer_spans = ["".join(a.split()) for a in answer_spans if len(a) > 0]
        else:
            answer_spans = [s.strip() for s in answer_spans if len(s) > 0]
        res.append(answer_spans)
    return res

def main():
    args = parse_args()

    with open(args.data_file, "r") as f:
        lines = [l.strip() for l in f]
        infos = []
        for line in tqdm(lines, mininterval=1):
            infos.append(json.loads(line)["translation"])

    answer_spans = parse_data_file(infos, args)

    pred_n_spans = []
    pred_1_spans = []
    for instance_i in range(len(answer_spans)):
        blank_num = len(answer_spans[instance_i])
        pred_n_spans.append([set() for _ in range(blank_num)])
        pred_1_spans.append([set() for _ in range(blank_num)])

    for d in args.preds:
        pred_fn = os.path.join(d, "answer_per_blank.txt")
        with open(pred_fn) as f:
            lines = [l.strip() for l in f]
        answer_preds = parse_answer_per_blank(lines)

        for instance_i in range(len(answer_preds)):
            for blank_i in range(len(answer_preds[instance_i])):
                if args.match_standard == "character":
                    cands = set(answer_preds[instance_i][blank_i])
                    pred_n_spans[instance_i][blank_i] = pred_n_spans[instance_i][blank_i].union(cands)
                    pred_1_spans[instance_i][blank_i].add(answer_preds[instance_i][blank_i][0])
                else:
                    cands = [c.lower() for c in answer_preds[instance_i][blank_i].strip().split("\t")]
                    pred_n_spans[instance_i][blank_i] = pred_n_spans[instance_i][blank_i].union(cands)
                    pred_1_spans[instance_i][blank_i].add(cands[0])

    total_cnt = match_1_cnt = match_n_cnt = 0
    for instance_i in range(len(answer_spans)):
        for blank_i in range(len(answer_spans[instance_i])):
            total_cnt += 1
            label = answer_spans[instance_i][blank_i].lower()
            if args.match_standard == "token":
                label = label.lower()
            top_n_preds = pred_n_spans[instance_i][blank_i]
            top_1_preds = pred_1_spans[instance_i][blank_i]
            if label in top_n_preds: match_n_cnt += 1
            if label in top_1_preds: match_1_cnt += 1

    msg = f"Jointly evaluated {', '.join(args.preds)}.\n"
    msg += f"Top 1 Match: {match_1_cnt / total_cnt:.4f} ({match_1_cnt} / {total_cnt})\n" \
           f"Top n Match: {match_n_cnt / total_cnt:.4f} ({match_n_cnt} / {total_cnt})\n"

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "match_rate.txt"), "w") as writer:
        writer.write(msg)

    print(msg)

if __name__ == '__main__':
    main()