#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', type=Path,
                        help='Path to the source corpus.')
    parser.add_argument('-mt', type=Path,
                        help='Path to the MT corpus.')
    parser.add_argument('-a', '--psd-align', type=Path,
                        help='Path to the pseudo alignment.')
    parser.add_argument('--source-tags', type=Path,
                        help='Path to the source tags file.')
    parser.add_argument('--mt-tags', type=Path,
                        help='Path to the MT tags file')

    args = parser.parse_args()

    return args


def output_res(res_lines):

    res_container = collections.defaultdict(int)
    for res in res_lines:
        for key in res.keys():
            res_container[key] += res[key]

    for k, v in res_container:
        print(f'{k}: {v}')
        

def main():
    args = parse_args()

    def read_tokens(fn):
        with fn.open() as f:
            return [l.strip().split() for l in f]

    src_tokens_lines = read_tokens(args.src)
    mt_tokens_lines = read_tokens(args.mt)
    aligns_lines = read_tokens(args.psd_align)
    source_tags_lines = read_tokens(args.source_tags)
    mt_tags_lines = read_tokens(args.mt_tokens)

    assert len(src_tokens_lines) == len(mt_tokens_lines) == len(aligns_lines) == len(source_tags_lines) == len(mt_tags_lines)

    res_lines = []
    for src_tokens, mt_tokens, aligns, source_tags, mt_tags in zip(src_tokens_lines, mt_tokens_lines, aligns_lines,
                                                                   source_tags_lines, mt_tags_lines):
        if len(mt_tags) == 2 * len(mt_tokens) + 1:
            mt_tags = mt_tags[1::2]

        align_map = collections.defaultdict(list)
        rev_align_map = collections.defaultdict(list)
        for a in aligns:
            i, j = map(int, a.split('-'))
            align_map[i].append(j)
            rev_align_map[j].append(i)

        res = collections.defaultdict(int)
        res['total_align'] = len(aligns)
        res['total_word'] = len(src_tokens) + len(mt_tokens)
        for i, src_tok in enumerate(src_tokens):
            if len(align_map[i]) == 0 and source_tags[i] != 'BAD':
                res['non_align_ok'] += 1
            elif len(align_map[i]) > 0:
                for j in align_map[i]:
                    if mt_tags[j] != source_tags[i]:
                        res['unmatch_tag'] += 1

        for j, mt_tok in enumerate(mt_tokens):
            if len(rev_align_map[j]) == 0 and mt_tags[j] != 'BAD':
                res['non_align_ok'] += 1

        res_lines.append(res)


    output_res(res_lines)

if __name__ == '__main__':
    main()