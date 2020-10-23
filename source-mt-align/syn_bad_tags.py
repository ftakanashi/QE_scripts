#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
import os
import numpy as np
import random
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', type=Path,
                        help='Path to the source corpus.')
    parser.add_argument('-t', '--tgt', type=Path,
                        help='Path to the target corpus.')
    parser.add_argument('-a', '--align', type=Path,
                        help='Path to the alignment file.')
    parser.add_argument('-st', '--source_tags', type=Path,
                        help='Path to the source tags file.')
    parser.add_argument('-tt', '--target_tags', type=Path,
                        help='Path to the target tags file.')

    parser.add_argument('--output_dir', type=Path,
                        help='Path to the output directory.')

    parser.add_argument('--replace_prob', type=float,
                        help='Probability to perform replacement upon an alignment. Keep the alignment but replace '
                             'the source and target words with random ones and set tags to BAD.')
    parser.add_argument('--drop_align_prob', type=float,
                        help='Probability to perform alignment dropout. Delete the alignment and set tags of both '
                             'source and target words to BAD if the alignment is the only one they have.')

    args = parser.parse_args()

    return args

def get_align_dict(aligns, reverse=False):
    d = collections.defaultdict(list)
    if type(aligns) is not list:
        aligns = aligns.split()
    for align in aligns:
        a, b = map(int, align.split('-'))
        if reverse:
            d[b].append(a)
        else:
            d[a].append(b)

    return d

def operate(args, src_tokens, tgt_tokens, aligns, src_tags, tgt_tags):

    operations = ['null', 'replace', 'drop_align']
    probs = np.array([1 - args.replace_prob - args.drop_align_prob, args.replace_prob, args.drop_align_prob])

    new_aligns = []
    for align in aligns:
        i, j = map(int, align.split('-'))
        if src_tags[i] != 'OK' or tgt_tags[j] != 'OK':
            new_aligns.append(align)
            continue

        flag = True
        operation = np.random.choice(operations, p=probs)
        if operation == 'null':
            pass

        elif operation == 'replace':
            new_token = random.choice(tgt_tokens)
            while new_token == tgt_tokens[j]:
                new_token = random.choice(tgt_tokens)
            tgt_tokens[j] = new_token
            src_tags[i] = tgt_tags[j] = 'BAD'

        elif operation == 'drop_align':
            align_dict = get_align_dict(aligns)
            rev_align_dict = get_align_dict(aligns, reverse=True)
            if len(align_dict[i]) == 0:
                src_tags[i] = 'BAD'
            if len(rev_align_dict[j]) == 0:
                tgt_tags[j] = 'BAD'
            flag = False

        else:
            raise ValueError(f'Invalid operation {operation}')

        if flag:
            new_aligns.append(align)

    return new_aligns

def main():
    args = parse_args()

    def read_fn(fn):
        with fn.open() as f:
            return [l.strip() for l in f]

    src_lines = read_fn(args.src)
    tgt_lines = read_fn(args.tgt)
    align_lines = read_fn(args.align)
    src_tags_lines = read_fn(args.source_tags)
    tgt_tags_lines = read_fn(args.target_tags)

    res_container = []
    for src_line, tgt_line, align_line, src_tags_line, tgt_tags_line in \
        zip(src_lines, tgt_lines, align_lines, src_tags_lines, tgt_tags_lines):

        src_tokens = src_line.split()
        tgt_tokens = tgt_line.split()
        aligns = align_line.split()
        src_tags = src_tags_line.split()
        tgt_tags = tgt_tags_line.split()
        assert len(src_tokens) == len(src_tags)
        assert len(tgt_tokens) == len(tgt_tags)

        new_aligns = operate(args, src_tokens, tgt_tokens, aligns, src_tags, tgt_tags)
        res_container.append([src_tokens, tgt_tokens, new_aligns, src_tags, tgt_tags])


    def wf_fn(fn):
        p, s = fn.rsplit('.', 1)
        return os.path.join(args.output_dir, f'{p}.syn.{s}')

    new_src = Path(wf_fn(args.src.name)).open('w')
    new_tgt = Path(wf_fn(args.tgt.name)).open('w')
    new_align = Path(wf_fn(args.align.name)).open('w')
    new_source_tags = Path(wf_fn(args.source_tags.name)).open('w')
    new_target_tags = Path(wf_fn(args.target_tags.name)).open('w')

    for res in res_container:
        src_tokens, tgt_tokens, aligns, src_tags, tgt_tags = res
        new_src.write(' '.join(src_tokens) + '\n')
        new_tgt.write(' '.join(tgt_tokens) + '\n')
        new_align.write(' '.join(aligns) + '\n')
        new_source_tags.write(' '.join(src_tags) + '\n')
        new_target_tags.write(' '.join(tgt_tags) + '\n')

    new_src.close()
    new_tgt.close()
    new_align.close()
    new_source_tags.close()
    new_target_tags.close()


if __name__ == '__main__':
    main()