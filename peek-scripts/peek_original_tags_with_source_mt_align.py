#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = '''
    Take a pair of source / MT corpus and their tags along with a source-MT alignment file.
    Analyze the content of the files, showing how many mismatch (OK-BAD or BAD-OK) tags are there and
    how many non-aligned OKs are there.
'''

import argparse
import collections
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-s', '--src', type=Path,
                        help='Path to the source corpus.')
    parser.add_argument('-t', '--tgt', type=Path,
                        help='Path to the MT corpus.')
    parser.add_argument('-a', '--source_mt_align', type=Path,
                        help='Path to the pseudo alignment.')
    parser.add_argument('-st', '--source_tags', type=Path,
                        help='Path to the source tags file.')
    parser.add_argument('-wt', '--mt_word_tags', type=Path,
                        help='Path to the MT tags file. Expect it to be MT-word-tags-only.')

    args = parser.parse_args()

    return args


def output_res(res_lines):

    res_container = collections.defaultdict(int)
    for res in res_lines:
        for key in res.keys():
            res_container[key] += res[key]

    for k in ('total_aligns', 'source_words', 'target_words',
              'source_non_aligned_OK', 'target_non_aligned_OK',
              'source_dismatched_OK', 'source_dismatched_BAD',
              'target_dismatched_OK', 'target_dismatched_BAD'):

        print(f'{k}: {res_container[k]}')

def main():
    args = parse_args()

    def read_tokens(fn):
        with fn.open() as f:
            return [l.strip().split() for l in f]

    src_tokens_lines = read_tokens(args.src)
    mt_tokens_lines = read_tokens(args.tgt)
    aligns_lines = read_tokens(args.source_mt_align)
    source_tags_lines = read_tokens(args.source_tags)
    mt_tags_lines = read_tokens(args.mt_word_tags)

    assert len(src_tokens_lines) == len(mt_tokens_lines) == len(aligns_lines) == len(source_tags_lines) == len(mt_tags_lines)

    res_lines = []
    for src_tokens, mt_tokens, aligns, source_tags, mt_tags in zip(src_tokens_lines, mt_tokens_lines, aligns_lines,
                                                                   source_tags_lines, mt_tags_lines):
        assert len(src_tokens) == len(source_tags)
        if len(mt_tags) == 2 * len(mt_tokens) + 1:
            import warnings
            warnings.warn('You provided full MT tags but only the MT word tags are used.')
            mt_tags = mt_tags[1::2]
        assert len(mt_tokens) == len(mt_tags)

        align_map = collections.defaultdict(list)
        rev_align_map = collections.defaultdict(list)
        for a in aligns:
            i, j = map(int, a.split('-'))
            align_map[i].append(j)
            rev_align_map[j].append(i)

        res = {
            'total_aligns': 0,
            'source_words': 0,
            'target_words': 0,
            'source_non_aligned_OK': 0,
            'target_non_aligned_OK': 0,
            'source_dismatched_OK': 0,
            'source_dismatched_BAD': 0,
            'target_dismatched_OK': 0,
            'target_dismatched_BAD': 0,
        }

        res['total_aligns'] = len(aligns)
        res['source_words'] = len(src_tokens)
        res['target_words'] = len(mt_tokens)
        for i, src_tok in enumerate(src_tokens):
            if len(align_map[i]) == 0 and source_tags[i] != 'BAD':
                res['source_non_aligned_OK'] += 1
            elif len(align_map[i]) > 0:
                for j in align_map[i]:
                    if mt_tags[j] != source_tags[i]:
                        res[f'source_dismatched_{source_tags[i]}'] += 1

        for j, mt_tok in enumerate(mt_tokens):
            if len(rev_align_map[j]) == 0 and mt_tags[j] != 'BAD':
                res['target_non_aligned_OK'] += 1
            elif len(rev_align_map[j]) > 0:
                for k in rev_align_map[j]:
                    if source_tags[k] != mt_tags[j]:
                        res[f'target_dismatched_{mt_tags[j]}'] += 1

        res_lines.append(res)

    output_res(res_lines)

if __name__ == '__main__':
    main()