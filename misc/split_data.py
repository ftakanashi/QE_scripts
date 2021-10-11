#!/usr/bin/env python

import argparse
import os
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_prefix', required=True,
                        help='Prefix of source files.')
    parser.add_argument('--tgt_prefix', nargs='+',
                        help='Prefixes of target files.')
    parser.add_argument('--tgt_ratio', nargs='+', type=float,
                        help='Ratio of all types of target files. Note that sum of all ratio must be 1.0 and numbers of '
                             'ratios here must equals to that of files specified by tgt_prefix.')

    parser.add_argument('--do_not_shuffle', action='store_true', default=False,
                        help='Do not shuffle the data before splitting.')


    args = parser.parse_args()

    assert len(args.tgt_prefix) == len(args.tgt_ratio), 'number of tgt_prefix must equals that of tgt_ratio'
    assert sum(args.tgt_ratio) == 1.0, 'Sum of tgt_ratio must be 1.0'
    return args

def main(args):

    def read_fn(fn):
        with open(fn) as f:
            lines = [l.strip() for l in f]
            return lines, len(lines)

    src_dir = os.path.dirname(os.path.abspath(args.src_prefix))   # raw/
    src_fn_prefix = os.path.basename(args.src_prefix)    # raw
    src_files_lines = {}
    std_len = None
    for f in os.listdir(src_dir):
        if f.startswith(src_fn_prefix):
            print(f'Reading {os.path.join(src_dir, f)}')
            lines, line_num = read_fn(os.path.join(src_dir, f))
            if std_len is None:
                std_len = line_num
            else:
                assert line_num == std_len, f'All source files must contain identical number of lines but {f} seems not'
            src_files_lines[f.lstrip(src_fn_prefix)] = lines

    indices = list(range(std_len))
    if not args.do_not_shuffle:
        random.shuffle(indices)

    for suffix, lines in src_files_lines.items():
        print(f'Writing {suffix} files...')
        head = tail = 0
        for prefix, ratio in zip(args.tgt_prefix, args.tgt_ratio):
            tail += int(std_len * ratio)
            part_indices = indices[head:tail]
            part_indices.sort()
            with open(prefix + suffix, 'w') as wf:
                for l_i in part_indices:
                    wf.write(lines[l_i] + '\n')
            head = tail


if __name__ == '__main__':
    args = parse_args()
    main(args)