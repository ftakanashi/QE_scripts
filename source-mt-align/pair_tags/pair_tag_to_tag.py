#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=Path,
                        help='Path to the input file.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Path to the output file.')
    parser.add_argument('-s', '--side', type=str, choices=['src', 'tgt'],
                        help='The side of the tags you want to extract.')

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    with args.input.open() as f:
        pair_tags_lines = [l.strip().split() for l in f]

    wf = args.output.open('w')
    for pair_tags_line in pair_tags_lines:
        tmp = []
        for pair_tag in pair_tags_line:
            if args.side == 'src':
                tag = pair_tag.split('-')[0]
            elif args.side == 'tgt':
                tag = pair_tag.split('-')[1]
            else:
                raise ValueError(f'Invalid side option {args.side}')

            if tag == 'NULL':    # 特殊处理
                tag = 'BAD'

            tmp.append(tag)

        wf.write(' '.join(tmp) + '\n')

    wf.close()

if __name__ == '__main__':
    main()