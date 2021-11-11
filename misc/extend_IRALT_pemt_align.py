#!/usr/bin/env python

import argparse
import collections

def search(pairs, target_pe):
    for i, (pe, mt) in enumerate(pairs):
        if pe == target_pe:
            return i
    return -1

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to the input file.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')

    parser.add_argument('-pe',
                        help='Path to the PE file.')
    parser.add_argument('-mt',
                        help='Path to the MT file.')

    args = parser.parse_args()
    return args

def main():
    def read_file(fn):
        with open(fn, 'r') as f:
            return [l.strip() for l in f]

    args = parse_args()

    pe_mt_lines = read_file(args.input)
    pe_lines = read_file(args.pe)
    mt_lines = read_file(args.mt)

    wf = open(args.output, 'w')

    for line_no in range(len(pe_mt_lines)):
        pe2mt_align_dict = collections.defaultdict(list)
        mt2pe_align_dict = collections.defaultdict(list)
        for align in pe_mt_lines[line_no].strip().split():
            a, b = map(int, align.split('-'))
            pe2mt_align_dict[a].append(b)
            mt2pe_align_dict[b].append(a)

        pe_token_len = len(pe_lines[line_no].strip().split())
        mt_token_len = len(mt_lines[line_no].strip().split())
        align_pairs = []
        for mt in range(mt_token_len):
            if len(mt2pe_align_dict[mt]) > 0:
                align_pairs.extend([(pe, mt) for pe in sorted(mt2pe_align_dict[mt])])
            else:
                align_pairs.append((None, mt))
        for pe in range(pe_token_len):
            if len(pe2mt_align_dict[pe]) == 0:
                if pe == 0: pos = 0
                else: pos = search(align_pairs, pe - 1) + 1
                align_pairs.insert(pos, (pe, None))

        align_pair_strs = []
        for a, b in align_pairs:
            sa = f"{a}" if a is not None else ''
            sb = f'{b}' if b is not None else ''
            align_pair_strs.append(f"{sa}-{sb}")
        wf.write(' '.join(align_pair_strs) + '\n')

    wf.close()

if __name__ == '__main__':
    main()