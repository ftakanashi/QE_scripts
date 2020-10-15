#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    This script implements mainly two functions.
    1. Modify the bad tags into fine-grained tags like replace tags, insert tags, delete tags, etc.
    2. Dropout the mis-match alignments which means two words in the alignment has opposite tags like OK-BAD.
'''

import argparse
import collections
import copy
import os

from pathlib import Path

OK_LABELS = ['OK']
BAD_LABELS = ['BAD', 'REP', 'INS', 'DEL']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-st', '--source_tags', type=Path,
                        help='Path to the source tag files.')
    parser.add_argument('-mt', '--mt_tags', type=Path,
                        help='Path to the MT tag files.\nNEEDS ONLY WORD TAG FILE!!!')
    parser.add_argument('-pa', '--pseudo_align', type=Path,
                        help='Path to the pseudo alignment file.')
    parser.add_argument('--output_dir', type=Path,
                        help='Path to the output directory.')

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


def modify_tags_and_aligns(source_tags, mt_tags, align_dict, reversed_align_dict):
    # the source word is not translated, thus INSERT is required
    for from_i, source_tag in enumerate(source_tags):
        if len(align_dict[from_i]) == 0 and source_tag == 'BAD':
            source_tags[from_i] = 'INS'

    # the MT word is redundant, thus DELETE is required
    for to_j, mt_tag in enumerate(mt_tags):
        if len(reversed_align_dict[to_j]) == 0 and mt_tag == 'BAD':
            mt_tags[to_j] = 'DEL'

    for from_i, source_tag in enumerate(source_tags):
        dummy_to_js = copy.deepcopy(align_dict[from_i])
        for to_j in dummy_to_js:
            mt_tag = mt_tags[to_j]

            if source_tag == 'OK' and mt_tag == 'OK':
                continue

            elif (source_tag in BAD_LABELS and mt_tag in BAD_LABELS):
                source_tags[from_i] = 'REP'
                mt_tags[to_j] = 'REP'

            elif (source_tag == 'OK' and mt_tag in BAD_LABELS):
                align_dict[from_i].remove(to_j)
                reversed_align_dict[to_j].remove(from_i)
                if len(reversed_align_dict[to_j]) == 0:
                    # when there is no available alignment rest for this MT word
                    mt_tags[to_j] = 'DEL'

            elif (source_tag in BAD_LABELS and mt_tag == 'OK'):
                align_dict[from_i].remove(to_j)
                reversed_align_dict[to_j].remove(from_i)
                if len(align_dict[from_i]) == 0:
                    # when there is no available alignment rest for this source word
                    source_tags[from_i] = 'INS'

            else:
                raise RuntimeError('Not recognized situation.')


def main():
    args = parse_args()

    def read_fn(fn):
        with fn.open() as f:
            return [l.strip() for l in f]

    source_tag_lines = read_fn(args.source_tags)
    mt_tag_lines = read_fn(args.mt_tags)
    align_lines = read_fn(args.pseudo_align)
    align_dicts = align_lines_to_dict(align_lines)

    new_source_tags, new_mt_tags, new_align_dicts = [], [], []
    for source_tag_line, mt_tag_line, align_dict in zip(source_tag_lines, mt_tag_lines, align_dicts):
        source_tags = source_tag_line.split()
        mt_tags = mt_tag_line.split()
        reversed_align_dict = reverse_align_dict(align_dict)

        max_align_si = max(align_dict.keys())
        max_align_ti = max(reversed_align_dict.keys())
        assert max_align_si + 1 <= len(source_tags)
        assert max_align_ti + 1 <= len(mt_tags)

        modify_tags_and_aligns(source_tags, mt_tags, align_dict, reversed_align_dict)

        new_source_tags.append(source_tags)
        new_mt_tags.append(mt_tags)
        new_align_dicts.append(align_dict)

    def get_new_fn(fn):
        fn = os.path.basename(fn)
        prefix, ext = fn.rsplit('.', 1)
        return os.path.join(args.output_dir, f'{prefix}.mod.{ext}')

    # write in new source tags
    with open(get_new_fn(args.source_tags), 'w') as f:
        for tags in new_source_tags:
            f.write(' '.join(tags) + '\n')

    # write in new mt tags
    with open(get_new_fn(args.mt_tags), 'w') as f:
        for tags in new_mt_tags:
            f.write(' '.join(tags) + '\n')

    # write in new alignment
    with open(get_new_fn(args.pseudo_align), 'w') as f:
        for align_dict in new_align_dicts:
            a = []
            for from_i in sorted(align_dict):
                for to_j in sorted(align_dict[from_i]):
                    a.append(f'{from_i}-{to_j}')
            f.write(' '.join(a) + '\n')


if __name__ == '__main__':
    main()
