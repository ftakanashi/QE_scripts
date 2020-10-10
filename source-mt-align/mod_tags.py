#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-st', '--source_tags', type=Path,
                        help='Path to the source tag files.')
    parser.add_argument('-mt', '--mt_tags', type=Path,
                        help='Path to the MT tag files.')
    parser.add_argument('-pa', '--pseudo_align', type=Path,
                        help='Path to the pseudo alignment file.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Path to the output file.')

    args = parser.parse_args()

    return args


def align_lines_to_dict(align_lines):
    res = []
    for align_line in align_lines:
        align_dict = collections.defaultdict(list)
        for align in align_line.split():
            x, y = map(int, align.split('-'))
            align_dict[x].append(y)

        # yield align_dict
        res.append(align_dict)

    return res


def reverse_align_dict(align_dict):
    reverse_dict = collections.defaultdict(list)
    for from_i in align_dict:
        for to_j in align_dict[from_i]:
            reverse_dict[to_j].append(from_i)

    return reverse_dict


def main():
    args = parse_args()

    def read_fn(fn):
        with fn.open() as f:
            return [l.strip() for l in f]

    source_tag_lines = read_fn(args.source_tags)
    mt_tag_lines = read_fn(args.mt_tags)
    align_lines = read_fn(args.pseudo_align)
    align_dicts = align_lines_to_dict(align_lines)

    for source_tag_line, mt_tag_line, align_dict in zip(source_tag_lines, mt_tag_lines, align_dicts):
        source_tags = source_tag_line.split()
        mt_tags = mt_tag_line.split()
        reversed_align_dict = reverse_align_dict(align_dict)

        new_source_tags = [None] * len(source_tags)
        new_mt_tags = [None] * len(mt_tags)
        for from_i, source_tag in enumerate(source_tags):
            for to_j in align_dict[from_i]:
                mt_tag = mt_tags[to_j]
                n_source_tag = new_source_tags[from_i]
                n_mt_tag = new_mt_tags[to_j]
                # if source_tag == mt_tag == 'OK':
                #     if n_source_tag.startswith('BAD')
