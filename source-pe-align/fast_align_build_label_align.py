#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', type=Path,
                        help='Path to the source corpus file.')
    parser.add_argument('-t', '--tgt', type=Path,
                        help='Path to the target corpus file.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Path to the output alignment file.')

    parser.add_argument('--fast-align-bin-dir', default='fast_align/build',
                        help='Directory where binary files fast_align and atools are saved.')
    parser.add_argument('--tmp-working-dir', default='tmp_make_align',
                        help='A directory where the middle results files are saved.')

    args = parser.parse_args()
    return args

def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

def main():
    args = parse_args()

    if not os.path.isdir(args.tmp_working_dir):
        os.makedirs(args.tmp_working_dir, exist_ok=True)

    concat_corpus = os.path.join(args.tmp_working_dir, 'concat_corpus')
    cmd = "paste " + str(args.src) + " " + str(args.tgt) + " | awk -F '\\t' '{print $1 \" ||| \" $2}' > " + \
          concat_corpus
    run_cmd(cmd)

    forward_align_fn = os.path.join(args.tmp_working_dir, 'forward.align')
    reverse_align_fn = os.path.join(args.tmp_working_dir, 'reverse.align')

    run_cmd(f'{args.fast_align_bin_dir}/fast_align -i {concat_corpus} -d -o -v > {forward_align_fn}')

    run_cmd(f'{args.fast_align_bin_dir}/fast_align -i {concat_corpus} -d -o -v -r > {reverse_align_fn}')

    run_cmd(f'{args.fast_align_bin_dir}/atools -i {forward_align_fn} -j {reverse_align_fn} -c grow-diag-final-and >'
            f' {args.output}')

if __name__ == '__main__':
    main()