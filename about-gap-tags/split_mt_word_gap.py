#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--tag-file')

    parser.add_argument('-wo', '--word-output-file')
    parser.add_argument('-go', '--gap-output-file')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    word_tags = []
    gap_tags = []

    with Path(args.tag_file).open() as f:
        print('Reading tag file...')
        old_tags = [l.strip().split() for l in f]

    for i in tqdm(range(len(old_tags)), mininterval=1.0):
        row_old_tags = old_tags[i]
        gap_tags.append(row_old_tags[0::2])
        word_tags.append(row_old_tags[1::2])

    with Path(args.word_output_file).open('w') as f:
        print('Writing word tags file.')
        for row_word_tags in word_tags:
            f.write(' '.join(row_word_tags) + '\n')

    with Path(args.gap_output_file).open('w') as f:
        print('Writing gap tags file.')
        for row_gap_tags in gap_tags:
            f.write(' '.join(row_gap_tags) + '\n')

if __name__ == '__main__':
    main()