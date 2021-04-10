#!/usr/bin/env python

import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src',
                        help='Path to the source corpus.')
    parser.add_argument('-t', '--tgt',
                        help='Path to the target corpus.')
    parser.add_argument('-a', '--align',
                        help='Path to the alignment file.')
    parser.add_argument('-o', '--output_dir',
                        help='Path to the output directory.')
    parser.add_argument('--sample_size', type=int, default=300,
                        help='Sample size.')

    parser.add_argument('--ignore_empty_align', action='store_true',
                        help='Set this flag to ignore empty alignment lines while sampling.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    def open_fn(fn):
        with open(fn) as f: return f.readlines()

    src_lines = open_fn(args.src)
    tgt_lines = open_fn(args.tgt)
    align_lines = open_fn(args.align)
    assert len(src_lines) == len(tgt_lines) == len(align_lines), 'Unmatched number of lines.'
    std_len = len(src_lines)

    if args.ignore_empty_align:
        i_range = [i for i in range(std_len) if align_lines[i].strip() != '']
    else:
        i_range = list(range(std_len))
    assert len(i_range) > 0, 'No valid align lines.'

    indices = random.sample(i_range, args.sample_size)
    indices.sort()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    src_wf = open(os.path.join(args.output_dir, os.path.basename(args.src)), 'w')
    tgt_wf = open(os.path.join(args.output_dir, os.path.basename(args.tgt)), 'w')
    ali_wf = open(os.path.join(args.output_dir, os.path.basename(args.align)), 'w')

    for i in indices:
        src_wf.write(src_lines[i].strip() + '\n')
        tgt_wf.write(tgt_lines[i].strip() + '\n')
        ali_wf.write(align_lines[i].strip() + '\n')

    src_wf.close(), tgt_wf.close(), ali_wf.close()

if __name__ == '__main__':
    main()