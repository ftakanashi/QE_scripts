#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
import os
import numpy as np

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=Path,
                        help='Path to the input file where probabilities for all tags are saved.')

    parser.add_argument('--split_prob_only', action='store_true',
                        help='Set the flag to generate the splited fileds for probability only.')
    parser.add_argument('--output_dir', default=None,
                        help='Path to the output directory.')


    args = parser.parse_args()

    return args

def split_probs(input_lines, args):
    res_container = collections.defaultdict(list)
    for l in input_lines:
        prob_tuples = [[float(f) for f in t.split('|')] for t in l.split()]

        res_container['ok'].append([t[0] for t in prob_tuples])
        res_container['rep'].append([t[1] for t in prob_tuples])
        res_container['ins'].append([t[2] for t in prob_tuples])
        res_container['del'].append([t[3] for t in prob_tuples])

    return res_container

def process(res_container, args):
    ok_all_probs = res_container['ok']
    rep_all_probs = res_container['rep']
    ins_all_probs = res_container['ins']
    del_all_probs = res_container['del']

    res = []
    for row_probs in zip(ok_all_probs, rep_all_probs, ins_all_probs, del_all_probs):
        row_res = []
        for ok_prob, rep_prob, ins_prob, del_prob in zip(*row_probs):
            # core of rules
            if ok_prob > args.ok_prob_threshold:
                tag = 'OK'
            else:
                bad_probs = np.array([rep_prob, ins_prob, del_prob])
                if bad_probs.std() < args.bad_prob_std_threshold:
                    tag = 'REP'
                else:
                    tag = ['REP', 'INS', 'DEL'][bad_probs.argmax()]

            row_res.append(tag)

        res.append(' '.join(row_res))

    assert args.input.endswith('.prob')
    wf = open(os.path.join(args.output_dir, args.input.strip('.prob')), 'w')
    for row in res:
        wf.write(row + '\n')


def main():
    args = parse_args()

    with args.input.open() as f:
        input_lines = [l.strip() for l in f]

    res_container = split_probs(input_lines, args)

    if args.split_prob_only:
        for suf in res_container:
            wf = open(os.path.join(args.output_dir, f'{args.input}.{suf}'), 'w')
            for l in res_container[suf]:
                wf.write(' '.join(str(f) for f in l) + '\n')
            wf.close()

        return

    process(res_container, args)


if __name__ == '__main__':
    main()