#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from pathlib import Path

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--word-tag-file', type=Path)
    parser.add_argument('-t', '--gap-tag-file', type=Path)

    parser.add_argument('-o', '--output', type=Path)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    def read_tag(fn):
        with Path(fn).open() as f:
            return [l.strip().split() for l in f]

    word_tags = read_tag(args.word_tag_file)
    gap_tags = read_tag(args.gap_tag_file)

    assert len(word_tags) == len(gap_tags)
    new_tags = []
    for i in range(len(word_tags)):
        row_word_tags = word_tags[i]
        row_gap_tags = gap_tags[i]
        assert len(row_word_tags) == len(row_gap_tags) - 1

        row_new_tags = []
        for j in range(len(row_word_tags)):
            row_new_tags.append(row_gap_tags[j])
            row_new_tags.append(row_word_tags[j])
        row_new_tags.append(row_gap_tags[-1])

        new_tags.append(row_new_tags)

    with Path(args.output).open('w') as f:
        for row in new_tags:
            f.write(' '.join(row) + '\n')


if __name__ == '__main__':
    main()