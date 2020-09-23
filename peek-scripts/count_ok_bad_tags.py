#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=Path, nargs='+',
                        help='Path to the input file.')

    args = parser.parse_args()
    return args

def count_one_file(fn, counter):
    with fn.open() as f:
        lines = [l.strip().split() for l in f]
        for l in lines:
            for t in l:
                counter[t] += 1

def main():
    args = parse_args()

    counter = collections.defaultdict(int)

    for f in args.input:
        count_one_file(f, counter)

    for k, v in sorted(counter.items()):
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()