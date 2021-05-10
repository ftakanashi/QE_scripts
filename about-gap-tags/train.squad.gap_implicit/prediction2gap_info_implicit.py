#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import collections
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prediction_file',
                        help='Path to the predictions_.json produced by SQuAD prediction model.')
    parser.add_argument('-to', '--tag_output',
                        help='Path to the gap tag output file. If a gap is not aligned to any source words, '
                             'it would be regarded as OK otherwise BAD.')

    parser.add_argument('-gao', '--src_gap_alignment_output', default=None,
                        help='Path to the src-gap alignment output file. Set it to None to avoid output. Default: None')
    # parser.add_argument('-wao', '--src_mt_alignment_output', default=None,
    #                     help='Path to the src-mt alignment output file. Set it to None to avoid output. Default: None')
    parser.add_argument('--src_gap_align_prob_threshold', default=0.4, type=float,
                        help='A probability threshold for extracting src-gap alignment. Default: 0.4')
    # parser.add_argument('--src_mt_align_prob_threshold', default=0.4, type=float,
    #                     help='A probability threshold for extracting src-mt alignment. Default: 0.4')

    args = parser.parse_args()
    return args


def parse_key(key):
    items = key.split('_')
    try:
        sent_id = int(items[0])
        word_id = int(items[1])
        items[0] = sent_id
        items[1] = word_id
    except ValueError as e:
        print('By default, assert first two flags are sentence id and word id.')
        raise e

    assert items[2] in ('s2t', 't2s'), 'By default, third flag should be direction that is either s2t or t2s'
    return tuple(items)


def process_one_row(row_res, args):
    pair2prob = collections.defaultdict(list)
    mt_len = -1
    for flags, v in row_res.items():
        word_id, direc = flags[1:3]
        if direc == 't2s': mt_len = max(mt_len, word_id + 1)
        if v == '': continue
        start_i, end_i, prob = v[3:6]

        if direc == 's2t':
            for t_i in range(start_i, end_i + 1):
                pair2prob[f'{word_id}-{t_i}'].append(prob)

        elif direc == 't2s':
            for s_i in range(start_i, end_i + 1):
                pair2prob[f'{s_i}-{word_id}'].append(prob)

        else:
            raise Exception(f'Invalid direction {direc}')

    src_gap_aligns, src_mt_aligns = [], []
    gap_tags = ['OK' for _ in range(mt_len + 1)]
    for pair_key, probs in pair2prob.items():
        s_i, t_i = map(int, pair_key.split('-'))
        if sum(probs) / 2 >= args.src_gap_align_prob_threshold:
            src_gap_aligns.append(pair_key)
            gap_tags[t_i] = 'BAD'
    return src_gap_aligns, src_mt_aligns, gap_tags


def main():
    args = parse_args()
    with open(args.prediction_file) as f:
        content = json.load(f)

    all_keys = sorted([parse_key(k) + (k, ) for k in content.keys()])
    i = cur_sent_id = 0
    tag_lines, src_gap_align_lines, src_mt_align_lines = [], [], []
    percent_base = 1
    while i < len(all_keys):

        row = {}
        while i < len(all_keys) and all_keys[i][0] == cur_sent_id:
            k = all_keys[i]
            row[k[:-1]] = content[k[-1]]
            i += 1

        src_gap_aligns, src_mt_aligns, gap_tags = process_one_row(row, args)
        src_gap_align_lines.append(' '.join(src_gap_aligns))
        src_mt_align_lines.append(' '.join(src_mt_aligns))
        tag_lines.append(' '.join(gap_tags))

        cur_sent_id += 1

        if i >= percent_base * (len(all_keys) // 10):
            print('{:.2f}% completed.'.format(i / len(all_keys) * 100))
            percent_base += 1

    with open(args.tag_output, 'w') as wf:
        for tag_line in tag_lines:
            wf.write(tag_line + '\n')

    if args.src_gap_alignment_output:
        with open(args.src_gap_alignment_output, 'w') as wf:
            for align_line in src_gap_align_lines:
                wf.write(align_line + '\n')
    # if args.src_mt_alignment_output:
    #     with open(args.src_mt_alignment_output, 'w') as wf:
    #         for align_line in src_mt_align_lines:
    #             wf.write(align_line + '\n')


if __name__ == '__main__':
    main()
