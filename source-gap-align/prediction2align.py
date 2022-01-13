#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Analyze a prediction.json produced by BERT
Transferring it into a alignment file format like 1-2 2-3 3-3...
Among one pair of sentences, an alignment a-b will be recorded only when
the average of p_{a-b} in s2t and p_{b-a} in t2s is greater than the threshold.
'''

import argparse
import collections
import json
import os

from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--mt",
                        help="Path to the MT corpus."
                             "It is used for referring to line number and token length.")

    parser.add_argument('-p', '--pred-json', default=None,
                        help='Path to prediction.json')

    parser.add_argument('-d', '--pred-output-dir', default=None,
                        help='Path to the output dir where output.N for shard N is saved.')

    parser.add_argument("-ao", "--align_output",
                        help="Path to the alignment output file.")
    parser.add_argument("-to", "--tag_output",
                        help="Path to the MT gap tag output file.")

    parser.add_argument("--prob-threshold", type=float, default=0.4,
                        help="Threshold for alignment extraction. Default: 0.4")

    opt = parser.parse_args()

    assert opt.pred_json is not None or opt.pred_output_dir is not None

    return opt

def get_info(data):
    '''
    extract information from prediction's json.
    for each sentence pair, a defaultdict with a default empty list will be used as a container to save the information
    about the alignments yield from both directions.
    Specifically, key is 'a-b' while a represents for an token index in source and b for target.
    Value of 'a-b' is a list containing probabilities fecthed from 'a_b_s2t' and 'b_a_t2s' items from the original
    json data.
    Note that the list shall never longer than 2.
    :param data:
    :return:
    '''
    info = {}
    for k in data:
        v = data[k]
        items = k.split("_")
        if len(items) == 3:
            sent_id, tok_id = map(int, items[:2])
            flag = items[2]
        elif len(items) == 4:
            sent_id, tok_id, span_id = map(int, items[:3])
            flag = items[3]
        else:
            raise ValueError(f"cannot analyze key: {k}")

        if sent_id not in info:
            info[sent_id] = collections.defaultdict(list)

        if v == '' : continue
        if v[3] == v[4] == -1: continue    # rare case adaption

        span_start, span_end = v[3], v[4]
        prob = v[5]
        if flag == "s2t":
            src_i = tok_id
            for tgt_i in range(span_start + 1, span_end + 1):
                info[sent_id][f"{src_i}-{tgt_i}"].append(prob)
        elif flag == "t2s":
            tgt_i = tok_id
            for src_i in range(span_start, span_end + 1):
                info[sent_id][f"{src_i}-{tgt_i}"].append(prob)
        else:
            raise ValueError(f"invalid direction flag: {flag}")

    return info

def process_one_line(sent_align_info, opt):
    '''
    iterate all possible pairs and only record those pairs with an average over threshold
    :param sent_align_info:
    :param opt:
    :return:
    '''

    alignments = []
    if len(sent_align_info) == 0:    # no alignment detected
        return alignments

    for k in sent_align_info.keys():
        probs = sent_align_info[k]
        try:
            assert len(probs) <= 2, f"find more than two probs for key: {k}"
            src_i = int(k.split("-")[0])
            tgt_i = int(k.split("-")[1]) * 2    # *2 because we want src-gap alignments
            if sum(probs) / 2 > opt.prob_threshold:
                alignments.append((src_i, tgt_i))

        except ValueError as e:
            raise ValueError(f"Error encountered while parsing key {k}")

    alignments.sort()
    return alignments

def main():
    opt = parse_args()

    info = {}

    if opt.pred_json is not None:
        with open(opt.pred_json, 'r') as f:
            data = json.loads(f.read())
        info = get_info(data)

    elif opt.pred_output_dir is not None:
        parent_dir = opt.pred_output_dir
        sub_dirs = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)]

        for sub_dir in sub_dirs:
            assert os.path.isfile(os.path.join(sub_dir, 'predictions.json')), \
            f'Cannot find prediction.json in {sub_dir}'

            print(f'Reading predictions.json in {sub_dir}...')
            with open(os.path.join(sub_dir, 'predictions.json'), 'r') as f:
                data = json.loads(f.read())
            sub_info = get_info(data)

            info.update(sub_info)

    with open(opt.mt) as f:
        mt_lines = [l.strip() for l in f]
    line_nums = len(mt_lines)

    align_wf = open(opt.align_output, "w")
    tag_wf =  open(opt.tag_output, "w")
    for sent_id in tqdm(range(line_nums), mininterval=1.0):
        mt_len = len(mt_lines[sent_id].strip().split())
        gap_tags = ["OK" for _ in range(mt_len + 1)]
        if sent_id not in info:
            aligns = []
        else:
            aligns = process_one_line(info[sent_id], opt)
            for _, gap_i in aligns:
                gap_tags[gap_i // 2] = "INS"

        align_wf.write(" ".join([f"{a}-{b}" for a, b in aligns]) + "\n")
        tag_wf.write(" ".join(gap_tags) + "\n")

if __name__ == '__main__':
    main()

