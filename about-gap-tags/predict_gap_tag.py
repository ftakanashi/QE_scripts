#!/usr/bin/env python
# -*- coding:utf-8 -*-
NOTE = \
'''
    A script designed for predicting the GAP QE Tags based on existing source word tags and MT word tags and the 
    pseudo alignments.
    NOTE. source corpus and MT corpus files technically are not needed in the program, but I added them for a simple 
    token-matching check by which you can assure that you are using the correct file.
'''

import argparse
import collections

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-s', '--src', type=Path,
                        help='Path to the source file.')
    parser.add_argument('-st', '--src-qe-tags', type=Path,
                        help='Path to the source QE tag file.')
    parser.add_argument('-t', '--translation', type=Path,
                        help='Path to the MT file.')
    parser.add_argument('-tt', '--mt-qe-tags', type=Path,
                        help='Path to the target QE tag file.')
    parser.add_argument('-a', '--alignment', type=Path,
                        help='Path to the Source-MT pseudo alignment file.')

    parser.add_argument('-o', '--output', type=Path,
                        help='Path to the output file.')

    args = parser.parse_args()

    return args


def process_one_pair(src_line, mt_line, src_qe_tags_line, mt_qe_tags_line, align_line):
    src_tokens = src_line.split()
    mt_tokens = mt_line.split()
    src_qe_tags = src_qe_tags_line.split()
    mt_qe_tags = mt_qe_tags_line.split()

    assert len(src_tokens) == len(src_qe_tags)
    assert len(mt_tokens) == len(mt_qe_tags)

    mt2src_align = collections.defaultdict(list)
    for align in align_line.split():
        i, j = map(int, align.split('-'))
        mt2src_align[j].append(i)

    gap_qe_tags = []


    ###############################
    #
    # Core: rules for judging
    #
    ###############################
    # first GAP
    if 0 in mt2src_align[0] or mt_qe_tags[0] == 'OK':
        gap_qe_tags.append('OK')
    else:
        gap_qe_tags.append('BAD')

    # middle GAPs
    for i in range(len(mt_tokens) - 1):
        left_align_to = mt2src_align[i]
        right_align_to = mt2src_align[i + 1]

        tag = 'BAD'
        if len(left_align_to) > 0 and len(right_align_to) > 0:
            for l in left_align_to:
                for r in right_align_to:
                    if abs(l - r) <= 1:
                        tag = 'OK'

        elif (len(left_align_to) > 0 and mt_qe_tags[i] == 'OK') or \
                (len(right_align_to) > 0 and mt_qe_tags[i + 1] == 'OK'):
            tag = 'OK'

        gap_qe_tags.append(tag)

    # last GAP
    last_src_i = len(src_tokens) - 1
    last_mt_i = len(mt_tokens) - 1
    if last_src_i in mt2src_align[last_mt_i] or mt_qe_tags[last_mt_i] == 'OK':
        gap_qe_tags.append('OK')
    else:
        gap_qe_tags.append('BAD')

    ###############################
    #
    # End of rules.
    #
    ###############################

    # merge the word tags and the gap tags
    extended_mt_qe_tags = []
    for i in range(len(mt_tokens)):
        extended_mt_qe_tags.append(gap_qe_tags[i])
        extended_mt_qe_tags.append(mt_qe_tags[i])
    extended_mt_qe_tags.append(gap_qe_tags[-1])

    return extended_mt_qe_tags


def main():
    args = parse_args()

    def read_f(path):
        with path.open() as f:
            return [l.strip() for l in f]

    src_lines = read_f(args.src)
    mt_lines = read_f(args.translation)
    src_qe_tags_lines = read_f(args.src_qe_tags)
    mt_qe_tags_lines = read_f(args.mt_qe_tags)
    align_lines = read_f(args.alignment)

    extended_mt_qe_tags_lines = []
    for data_group in zip(src_lines, mt_lines, src_qe_tags_lines, mt_qe_tags_lines, align_lines):
        extended_mt_qe_tags = process_one_pair(*data_group)
        extended_mt_qe_tags_lines.append(' '.join(extended_mt_qe_tags))

    with Path(args.output).open('w') as f:
        for l in extended_mt_qe_tags_lines:
            f.write(l + '\n')

if __name__ == '__main__':
    main()