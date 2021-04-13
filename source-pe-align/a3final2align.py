#!/usr/bin/env python
# -*- coding:utf-8 -*-

from pathlib import Path
import argparse
import collections

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to the A3.final file.')
    parser.add_argument('-o', '--output',
                        help='Path to the output alignment file.')
    parser.add_argument('--reverse', action='store_true', default=False,
                        help='Set this flag to do reverse transformation.')

    return parser.parse_args()

def a3final_to_align_lines(a3final_fn, write_tmp_fn, reverse=False):

    with Path(a3final_fn).open() as f:
        a3final_lines = [l.strip() for l in f if not l.startswith('#')]

    def analyze(line):
        tokens = line.split()
        tok_i, word_i = 0, -1  # -1 stands for NULL
        align_dict = collections.defaultdict(list)
        while tok_i < len(tokens):
            if tokens[tok_i] == '({':
                tok_i += 1
                while tokens[tok_i] != '})':
                    align_dict[word_i].append(int(tokens[tok_i]) - 1)
                    tok_i += 1
                word_i += 1
            tok_i += 1
        return align_dict

    align_lines = []
    for l in a3final_lines:
        if l.startswith('NULL ({'):
            align_dict = analyze(l)
            aligns = []
            for from_i in sorted(align_dict):
                if from_i < 0: continue
                to_is = align_dict[from_i]
                if reverse:
                    aligns.extend([f'{to_i}-{from_i}' for to_i in to_is])
                else:
                    aligns.extend([f'{from_i}-{to_i}' for to_i in to_is])
            align_lines.append(' '.join(aligns))

    with Path(write_tmp_fn).open('w') as f:
        for l in align_lines:
            f.write(l + '\n')

def main():
    args = parse_args()
    a3final_to_align_lines(args.input, args.output, reverse=args.reverse)

if __name__ == '__main__':
    main()