#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os

from pathlib import Path
from subprocess import Popen, PIPE


def run_cmd(cmd):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if err:
        raise RuntimeError(f'Error encountered while running [{cmd}]:\n{err}')
    else:
        return out


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


def main():
    args = parse_args()

    if not os.path.isdir(args.tmp_working_dir):
        os.makedirs(args.tmp_working_dir)

    concat_corpus = os.path.join(args.tmp_working_dir, 'concat_corpus')
    cmd = "paste " + args.src + " " + args.tgt + " > awk -F '\\t' '{print $1 \" ||| \" $2}' > " + concat_corpus
    run_cmd(cmd)

    forward_align_fn = os.path.join(args.tmp_working_dir, 'forward.align')
    reverse_align_fn = os.path.join(args.tmp_working_dir, 'reverse.align')
    print(
        run_cmd(f'{args.fast_align_bin_dir}/fast_align -i {concat_corpus} -d -o -v > {forward_align_fn}')
    )

    print(
        run_cmd(f'{args.fast_align_bin_dir}/fast_align -i {concat_corpus} -d -o -v -r > {reverse_align_fn}')
    )

    print(
        run_cmd(
            f'{args.fast_align_bin_dir}/atools -i {forward_align_fn} -j {reverse_align_fn} -c grow-diag-final-and > {args.output}')
    )

if __name__ == '__main__':
    main()