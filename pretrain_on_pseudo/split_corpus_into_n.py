#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src',
                        help='Path to the original source corpus.')
    parser.add_argument('-t', '--tgt',
                        help='Path to the original target corpus.')
    parser.add_argument('-n', '--split_num', type=int,
                        help='How many (N) splits do you want.')
    parser.add_argument('-op', '--output_dir_prefix',
                        help='Path to the output directory prefix.')

    args = parser.parse_args()

    assert args.split_num <= 30, 'It is an insurance. Please modify the code if you really want so many splits.'
    return args

def main():
    args = parse_args()

    def read_fn(fn):
        print(f'Reading {fn}...')
        with open(fn) as f:
            lines = []
            for l in tqdm(f, mininterval=1, ncols=50):
                lines.append(l.strip())
            return lines

    src_ext = os.path.splitext(args.src)[1]
    tgt_ext = os.path.splitext(args.tgt)[1]

    src_lines = read_fn(args.src)
    tgt_lines = read_fn(args.tgt)
    std_len = len(src_lines)
    assert len(tgt_lines) == std_len, 'Unmatched number of lines.'

    split_len = std_len // args.split_num

    for n in range(args.split_num):
        print(f'Start processing split No.{n}')
        start_i, end_i = n * split_len, (n+1) * split_len
        split_dir = args.output_dir_prefix + f'.{n}'
        os.makedirs(split_dir, exist_ok=False)

        join = os.path.join
        train_src = open(join(split_dir, f'train{src_ext}'), 'w')
        train_tgt = open(join(split_dir, f'train{tgt_ext}'), 'w')
        test_src = open(join(split_dir, f'test{src_ext}'), 'w')
        test_tgt = open(join(split_dir, f'test{tgt_ext}'), 'w')
        for i in tqdm(range(std_len), mininterval=1, ncols=50):
            if start_i <= i < end_i:
                test_src.write(src_lines[i] + '\n')
                test_tgt.write(tgt_lines[i] + '\n')
            else:
                train_src.write(src_lines[i] + '\n')
                train_tgt.write(tgt_lines[i] + '\n')

        for wf in (train_src, train_tgt, test_src, test_tgt):
            wf.close()

if __name__ == '__main__':
    main()