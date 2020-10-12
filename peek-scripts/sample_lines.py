#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import random

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--files', nargs='+',
                        help='List of files')
    parser.add_argument('-ln', '--line_num', nargs='+', type=int,
                        help='Specify the indices of lines to peek.')
    parser.add_argument('-lnf', '--line_num_file', type=Path, default=None,
                        help='A file contains line numbers splited with comma. The line numbers in the file will be '
                             'peeked.')
    parser.add_argument('-rs', '--random_sample', type=int, default=-1,
                        help='Specify the number of lines which will be randomly sampled from files.')

    args = parser.parse_args()

    assert len(args.files) > 0
    assert args.line_num is not None or args.random_sample is not None

    return args

def print_out(line_no, files_lines):
    print('=' * 10 + f' LINE_NO[{line_no}] ' + '=' * 10)
    for lines in files_lines:
        print(lines[line_no])
        print('')
    print('')

def main():
    args = parse_args()

    files_lines = []
    for f in args.files:
        with Path(f).open() as f:
            files_lines.append([l.strip() for l in f])

    std_len = len(files_lines[0])
    for lines in files_lines:
        assert len(lines) == std_len

    if args.line_num is not None:
        assert len(args.line_num) > 0
        sampled_indices = args.line_num

    elif args.random_sample > 0:
        sampled_indices = random.sample(range(std_len), args.random_sample)

    elif args.line_num_file is not None:
        with args.line_num_file.open() as f:
            sampled_indices = [int(i) for i in f.read().split(',')]

    for ln in sorted(sampled_indices):
        print_out(ln, files_lines)

if __name__ == '__main__':
    main()