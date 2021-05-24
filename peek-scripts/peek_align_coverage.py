#!/usr/bin/env python

import argparse
import collections

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', help='Path to the source corpus.')
    parser.add_argument('-t', '--tgt', help='Path to the target corpus.')
    parser.add_argument('-a', '--align', help='Path to the alignment file.')

    args = parser.parse_args()
    return args

def parse_alignment(align_line, rev=False):
    align_tokens = align_line.split()
    align_pairs = []
    for tok in align_tokens:
        a, b = tok.split('-')
        a, b = int(a), int(b)
        align_pairs.append((a, b))
    align_dict = collections.defaultdict(list)
    for a, b in align_pairs:
        if not rev:
            align_dict[a].append(b)
        else:
            align_dict[b].append(a)
    return align_dict

def main():
    args = parse_args()

    def read_fn(fn):
        print(f'Reading {fn}...')
        with open(fn) as f:
            return [l.strip() for l in f]

    src_lines, tgt_lines, align_lines = \
    read_fn(args.src), read_fn(args.tgt), read_fn(args.align)

    std_len = len(src_lines)
    for lines in (tgt_lines, align_lines):
        assert len(lines) == std_len, 'Unmatched number of lines.'

    src_non_aligned_count = tgt_non_aligned_count = 0
    src_total_count = tgt_total_count = 0
    for src_line, tgt_line, align_line in zip(src_lines, tgt_lines, align_lines):
        src_tokens = src_line.split()
        tgt_tokens = tgt_line.split()
        align_dict = parse_alignment(align_line)
        rev_align_dict = parse_alignment(align_line, rev=True)

        for i in range(len(src_tokens)):
            if len(align_dict[i]) == 0:
                src_non_aligned_count += 1
            src_total_count += 1
        for j in range(len(tgt_tokens)):
            if len(rev_align_dict[j]) == 0:
                tgt_non_aligned_count += 1
            tgt_total_count += 1

    print('')
    print('=' * 20)
    print(f'Source (Non-Aligned/Total): {src_non_aligned_count}/{src_total_count}')
    print(f'Target (Non-Aligned/Total): {tgt_non_aligned_count}/{tgt_total_count}')

if __name__ == '__main__':
    main()