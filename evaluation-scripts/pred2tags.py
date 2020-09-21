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
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='A threshold above which the corresponding tag will be recogized as BAD. Otherwise OK. '
                             'DEFAULT: 0.5')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    with args.input.open() as f:
        lines = [[float(t) for t in l.strip().split()] for l in f]

    new_lines = []
    for l in lines:
        new_l = ['BAD' if s > args.threshold else 'OK' for s in l]
        new_lines.append(' '.join(new_l))

    with args.output.open('w') as f:
        for line in new_lines:
            f.write(line + '\n')

if __name__ == '__main__':
    main()

