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

def count_one_file(fn):
    counter = collections.defaultdict(int)
    with fn.open() as f:
        lines = [l.strip().split() for l in f]
        for l in lines:
            for t in l:
                counter[t] += 1

    print(f'===== {fn} =====')
    for k,v in sorted(counter.items()):
        print(f'{k}: {v}')
    print('')
    return counter

def main():
    args = parse_args()

    total_counter = {}

    for f in args.input:
        counter = count_one_file(f)
        for k, v in counter.items():
            if k not in total_counter:
                total_counter[k] = v
            else:
                total_counter[k] += v
    print('\n')
    print('===== Total =====')
    for k, v in sorted(total_counter.items()):
        print(f'{k}: {v}')

if __name__ == '__main__':
    main()