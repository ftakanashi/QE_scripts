#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    THIS SCRIPT IS UNDER DEVELOPING
'''

import argparse
import collections
import os

from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--align', type=Path,
                        help='Path to the alignment file.')
    parser.add_argument('-st', '--source_tags', type=Path,
                        help='Path to the source tags file.')
    parser.add_argument('-tt', '--target_tags', type=Path,
                        help='Path to the target tags file.')

    parser.add_argument('--output_dir', type=Path,
                        help='Path to the output directory.')

    args = parser.parse_args()

    return args

def get_align_dict(aligns, reverse=False):
    if type(aligns) is str:
        aligns = aligns.split()

    d = collections.defaultdict(list)
    for a in aligns:
        i, j = map(int, a.split('-'))
        if reverse:
            d[j].append(i)
        else:
            d[i].append(j)

    return d

def generate_pair_tags(align_line, src_tags_line, tgt_tags_line):

    align_dict = get_align_dict(align_line)
    rev_align_dict = get_align_dict(align_line, reverse=True)
    src_tags = src_tags_line.split()
    tgt_tags = tgt_tags_line.split()
    src_pair_tags, tgt_pair_tags = [], []

    for i, src_tag in enumerate(src_tags):
        pair_tags = [f'{src_tag}-{tgt_tags[j]}' for j in align_dict[i]]
        if len(pair_tags) == 0:
            pair_tag = f'{src_tag}-NULL'
        else:
            pt_counter = collections.Counter(pair_tags)
            pair_tag = pt_counter.most_common()[0][0]
        src_pair_tags.append(pair_tag)

    for j, tgt_tag in enumerate(tgt_tags):
        pair_tags = [f'{tgt_tag}-{src_tags[i]}' for i in rev_align_dict[j]]
        if len(pair_tags) == 0:
            pair_tag = f'{tgt_tag}-NULL'
        else:
            pt_counter = collections.Counter(pair_tags)
            pair_tag = pt_counter.most_common()[0][0]
        tgt_pair_tags.append(pair_tag)

    return src_pair_tags, tgt_pair_tags





def main():
    args = parse_args()

    def read_fn(fn):
        with fn.open() as f:
            return [l.strip() for l in f]

    def wf_fn(fn):
        p, s = fn.rsplit('.', 1)
        return os.path.join(args.output_dir, f'{p}.pair.{s}')

    align_lines = read_fn(args.align)
    src_tags_lines = read_fn(args.source_tags)
    tgt_tags_lines = read_fn(args.target_tags)

    wf_src = Path(wf_fn(args.source_tags)).open('w')
    wf_tgt = Path(wf_fn(args.target_tags)).open('w')
    for align_line, src_tags_line, tgt_tags_line in zip(align_lines, src_tags_lines, tgt_tags_lines):
        src_pair_tags, tgt_pair_tags = generate_pair_tags(align_line, src_tags_line, tgt_tags_line)
        wf_src.write(' '.join(src_pair_tags))
        wf_tgt.write(' '.join(tgt_pair_tags))

if __name__ == '__main__':
    main()