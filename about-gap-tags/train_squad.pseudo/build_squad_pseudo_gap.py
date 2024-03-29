#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    This script takes a pair of source corpus and target corpus (PE or MT) as input.
    
    When target is PE, the script mainly generate the pseudo training data for gap-tag-prediction model.
    Concretely, based on the original PE corpus, two types of incorporation of GAP is exploited.
    Firstly, an original PE word will have a chance to be replaced by GAP. If it is aligned to some source words then 
    a src-gap alignment is built.
    Secondly, a gap between two original PE words also has a chance to be picked out. It will be regarded as a 
    non-aligned gap.
'''

import argparse
import collections
import json
import random
import sys

def parse_alignment(align_lines):
    '''
    Turn every raw alignment line into a defaultdict(list) where keys are src indexs and value_list containing tgt
    indexs.
    :return a list of defaultdict(list) corresponding to every line pair.
    '''
    res = []
    for align_line in align_lines:
        align_info = collections.defaultdict(list)
        aligns = align_line.strip().split()
        for align in aligns:
            s, t = map(int, align.split('-'))
            align_info[s].append(t)
        res.append(align_info)

    return res

def reverse_align_lines(align_lines):
    '''
    reverse every pair in a set of align_lines.
    e.g. ['1-2 2-3', '1-3 3-3']  -->  ['2-1 3-2', '3-1 3-3']
    '''
    res = []
    delimeters = ['p', '-']
    for align_line in align_lines:
        aligns = align_line.split()
        new_aligns = []
        for align in aligns:
            for d in delimeters:
                if d in align:
                    a, b = align.split(d)
                    new_aligns.append(f'{b}{d}{a}')
                    break

        res.append(' '.join(new_aligns))

    return res

def point2span(span):
    '''
    detect all spans in a list of non-continuous indices
    E.g. [1,2,4,5,6,8]  ->  [[1,2], [4,5,6], [8,]]
    '''
    if len(span) == 0: return []
    sorted_span = list(sorted(span))
    res = []
    tmp = [sorted_span[0], ]
    for cursor in sorted_span[1:]:
        if cursor - tmp[-1] == 1:
            tmp.append(cursor)
        else:
            res.append(tmp)
            tmp = [cursor, ]

    res.append(tmp)
    return res

def calc_answer_start(tgt_tokens, target_span):
    '''
    calculate the start index of answer in character unit.
    '''
    start_tok_i = min(target_span)
    total_len = start_tok_i  # length of spaces
    for t in tgt_tokens[:start_tok_i]:
        total_len += len(t)
    return total_len

def process_one_pair(src_line, tgt_line, s2t_align_info, t2s_align_info, pair_id, args):
    global possible_count
    global impossible_count

    src_tokens, tgt_tokens, gap_token, special_token = \
        src_line.split(), tgt_line.split(), args.gap_token, args.special_token

    # following probabilities will be set to
    sample_align_prob = args.sample_align_prob if s2t_align_info else 0.0
    add_all_mt_gaps = args.add_all_mt_gaps

    def make_aug_src(tokens, i, **kwargs):
        mode = kwargs.get('mode', 'add')
        if mode not in ('add', 'rep'): raise Exception(f'Invalid mode {mode}.')
        all_mt_gaps = add_all_mt_gaps == 1
        tmp_tokens = tokens.copy()
        if all_mt_gaps:
            gap_tokens = [gap_token for _ in range(len(tmp_tokens) + 1)]
            if mode == 'rep':
                tmp_tokens.pop(i)
                gap_tokens.pop()
            gap_tokens[i] = f'{special_token} {gap_token} {special_token}'
            res = []
            for j in range(len(tmp_tokens)):
                res.append(gap_tokens[j])
                res.append(tmp_tokens[j])
            res.append(gap_tokens[-1])
            return ' '.join(res)

        else:
            if mode == 'add': tmp_tokens.insert(i, f'{special_token} {gap_token} {special_token}')
            elif mode == 'rep': tmp_tokens[i] = f'{special_token} {gap_token} {special_token}'
            return ' '.join(tmp_tokens)

    t2s_res = {'context': src_line, 'qas': []}

    if t2s_align_info:
        for i, tgt_token in enumerate(tgt_tokens):
            if random.random() > sample_align_prob or i not in t2s_align_info: continue

            item = {}
            item['id'] = f'{pair_id}_{i}_t2s_rep'
            item['question'] = make_aug_src(tgt_tokens, i, mode='rep')

            spans = point2span(t2s_align_info[i])
            if len(spans) > 0:
                item['is_impossible'] = False
                possible_count += 1
                answers = [
                    {
                        'answer_start': calc_answer_start(src_tokens, span),
                        'text': ' '.join([src_tokens[t] for t in span])
                    }
                    for span in spans
                ]
            else:
                item['is_impossible'] = True
                impossible_count += 1
                answers = [{'answer_start': -1, 'text': ''}]
            item['answers'] = answers

            t2s_res['qas'].append(item)

    # no matter alignment is provided, we always do add_gap data generation.
    # the difference is that if alignment is not provided, we generate testing data by default. So all gaps will be
    # generated once so the add_gap_prob is set to 1.0 automatically.
    for i in range(len(tgt_tokens) + 1):
        impossible_count += 1
        item = {}
        item['id'] = f'{pair_id}_{i}_t2s_add'
        item['is_impossible'] = True
        item['question'] = make_aug_src(tgt_tokens, i, mode='add')
        item['answers'] = [{'answer_start': -1, 'text': ''}]

        t2s_res['qas'].append(item)

    return t2s_res

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', help='Path to the source file.')
    parser.add_argument('-t', '--tgt', help='Path to the PE(for pseudo data generation) / MT(for real gap data '
                                            'generation) file.')
    parser.add_argument('-o', '--output', help='Path to the output file.')
    parser.add_argument('-a', '--align', default=None, help='Path to the Source-PE alignment file if it is needed.')

    parser.add_argument('--sample_align_prob', type=float, default=0.75,
                        help='The probability to sample a SRC-PE pair for pseudo src-gap data generation. Only valid '
                             'when -a is specified (means that you are generating training data). Default: 0.75')
    parser.add_argument('--add_all_mt_gaps', type=int, default=1, choices=[0,1],
                        help='A flag deciding whether add all gaps into the target side while genrating SQuAD data. '
                             'Default: 1. Set it to 0 to turn down this switch.')
    parser.add_argument('--gap_token', default='[GAP]', help='the token representing gaps. Default: [GAP]')
    parser.add_argument('--special_token', default='¶', help='a special token marking out corresponding word. '
                                                             'Default: ¶')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    def read_fn(fn):
        with open(fn, 'r') as f:
            return [l.strip() for l in f]

    src_lines = read_fn(args.src)
    tgt_lines = read_fn(args.tgt)

    std_len = len(src_lines)
    assert len(tgt_lines) == std_len, 'Number of lines does not match for SRC and PE.'

    if args.align:
        if args.tgt.endswith('.mt'):
            flag = input('You specified the alignment but looks like that you are inputting MT file as the target '
                         'corpus. Do you want to quit? (y/n)')
            if flag == 'y':
                sys.exit(1)

        align_lines = read_fn(args.align)
        assert len(align_lines) == std_len, 'Number of lines does not match for SRC and Alignment'
        s2t_align_infos = parse_alignment(align_lines)
        t2s_align_infos = parse_alignment(reverse_align_lines(align_lines))

    else:
        if args.tgt.endswith('.pe'):
            flag = input('You didn\'t specify the alignment but looks like that you are inputting PE file as the '
                         'target corpus. Do you want to quit? (y/n)')
            if flag == 'y':
                sys.exit(1)
        s2t_align_infos = t2s_align_infos = [None for _ in range(std_len)]

    global possible_count
    global impossible_count
    possible_count = impossible_count = 0

    total_res = {'version': 'v2.0'}
    data = []
    total_res['data'] = data

    pair_id = 0
    for src_line, tgt_line, s2t_align_info, t2s_align_info in \
            zip(src_lines, tgt_lines, s2t_align_infos, t2s_align_infos):
        pair_t2s_res = process_one_pair(src_line, tgt_line, s2t_align_info, t2s_align_info, pair_id, args)

        data.append({
            'paragraphs': [pair_t2s_res, ],
            'title': f'{pair_id}'
        })

        pair_id += 1

    content = json.dumps(total_res, indent=4)
    with open(args.output, 'w') as wf:
        wf.write(content)

    print(f'{possible_count} answerable records and {impossible_count} non-answerable records are written.')


if __name__ == '__main__':
    main()