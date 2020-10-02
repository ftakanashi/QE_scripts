#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', type=Path,
                        help='Path to the source corpus ** WITH BPE **.')
    parser.add_argument('-t', '--target', type=Path,
                        help='Path to the target corpus ** WITH BPE **.')
    parser.add_argument('-a', '--align', type=Path,
                        help='Path to the alignment file which corresponds to data in BPE format.')

    args = parser.parse_args()

    return args

def align_str_lines_to_dict(align_lines):
    res = []
    for align_line in align_lines:
        align_dict = collections.defaultdict(list)
        aligns = align_line.split()
        for a in aligns:
            x, y = map(int, a.split('-'))
            align_dict[x].append(y)

        res.append(align_dict)
    return res

def main():
    args = parse_args()

    with args.source.open() as f:
        src_lines = [l.strip() for l in f]

    with args.target.open() as f:
        tgt_lines = [l.strip() for l in f]

    with args.align.open() as f:
        align_lines = [l.strip() for l in f]

    align_dicts = align_str_lines_to_dict(align_lines)

    
