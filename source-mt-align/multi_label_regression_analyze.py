#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os

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
    res_container = [list() for _ in range(4)]
    for l in input_lines:
        probs_container = [list() for _ in range(4)]
        for t in l.split():
            probs = t.split('|')
            for i in range(4):
                probs_container[i].append(probs[i])
        for i in range(4):
            res_container[i].append(' '.join(probs_container[i]))

    for i, suf in enumerate(('ok', 'rep', 'ins', 'del')):
        wf = open(os.path.join(args.output_dir, f'{args.input}.{suf}'), 'w')
        for l in res_container[i]:
            wf.write(l + '\n')
        wf.close()

def main():
    args = parse_args()

    with args.input.open() as f:
        input_lines = [l.strip() for l in f]

    if args.split_prob_only:
        split_probs(input_lines, args)


if __name__ == '__main__':
    main()