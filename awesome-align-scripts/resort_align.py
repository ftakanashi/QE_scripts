#!/usr/bin/env python

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to the input file.')
    parser.add_argument('-o', '--output', default=None,
                        help='Path to the output file. If not specified, modify the input file in place.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.input, 'r') as f:
        align_lines = [l.strip() for l in f]

    new_lines = []
    for align_line in align_lines:
        align_pairs = align_line.split()
        aligns = []
        for p in align_pairs:
            a, b = map(int, p.split('-'))
            aligns.append((a, b))
        aligns.sort()
        new_lines.append(' '.join([f'{i}-{j}' for i, j in aligns]))

    wf = open(args.output if args.output else args.input, 'w')
    for l in new_lines:
        wf.write(l + '\n')
    wf.close()

if __name__ == '__main__':
    main()