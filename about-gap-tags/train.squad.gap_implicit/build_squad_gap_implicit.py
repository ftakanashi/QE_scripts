#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import bisect
import collections
import json

from typing import *


def parse_alignment(align_lines):
    '''
    Turn every raw alignment line into a defaultdict(list) where keys are src indexs and value_list containing tgt
    indexs.
    :return a list of defaultdict(list) corresponding to every line pair.
    '''
    res = []
    for align_line in align_lines:
        if align_line is None:
            res.append(None)
            continue
        align_info = collections.defaultdict(list)
        aligns = align_line.strip().split()
        for align in aligns:
            s, t = map(int, align.split('-'))
            align_info[s].append(t)
        res.append(align_info)

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


def get_answer_info(context: str, start_tok_i: int, end_tok_i: int) -> Dict:
    context_tokens = context.strip().split()
    assert len(context) == len(' '.join(context_tokens))

    answer_start = 0
    answer_span = []
    for i in range(start_tok_i):
        answer_start += (len(context_tokens[i]) + 1)

    if start_tok_i == len(context_tokens):
        answer_text = ''
    else:
        for i in range(start_tok_i, end_tok_i + 1):
            answer_span.append(context_tokens[i])
        answer_text = ' '.join(answer_span)

    return {'answer_start': answer_start, 'text': answer_text}


def reverse_align(forward_align):
    reversed_align = collections.defaultdict(list)
    for s_i, t_is in forward_align.items():
        for t_i in t_is:
            pos = bisect.bisect_left(reversed_align[t_i], s_i)
            reversed_align[t_i].insert(pos, s_i)
    return reversed_align


def process_one_pair(pair_id, src_line, mt_line, src_gap_align, src_mt_align, args):
    global possible_count
    global impossible_count

    # gap_token = args.gap_token
    mark_token = args.special_token

    src_tokens = src_line.strip().split()
    src_line = ' '.join(src_tokens)
    mt_tokens = mt_line.strip().split()
    mt_line = ' '.join(mt_tokens)

    # # explicitly insert gaps into MT
    # gapped_mt_tokens = []
    # for i in range(len(mt_tokens)):
    #     gapped_mt_tokens.append(gap_token)
    #     gapped_mt_tokens.append(mt_tokens[i])
    # gapped_mt_tokens.append(gap_token)
    # gapped_mt_line = ' '.join(gapped_mt_tokens)

    gap_src_align = reverse_align(src_gap_align) if src_gap_align is not None else None
    mt_src_align = reverse_align(src_mt_align) if src_mt_align is not None else None

    def generate_quesiton_str(tokens, i):
        tmp = tokens.copy()
        if i >= len(tmp):
            tmp.append(mark_token)
            tmp.append(mark_token)
        else:
            tmp.insert(i + 1, mark_token)
            tmp.insert(i, mark_token)
        return ' '.join(tmp)

    # collect s2t qas
    s2t_qas = []
    s2t_res = {
        'paragraphs': [{
            'context': mt_line,
            'qas': s2t_qas
        }],
        'title': f'{pair_id}_s2t'
    }
    for tok_id, token in enumerate(src_tokens):
        question = generate_quesiton_str(src_tokens, tok_id)
        empty_record = {
            'id': f'{pair_id}_{tok_id}_s2t',
            'question': question,
            'is_impossible': True,
            'answers': [{'answer_start': -1, 'text': ''}]
        }
        if src_gap_align is not None:
            if len(src_gap_align[tok_id]) > 0:
                for t_i in src_gap_align[tok_id]:
                    mapped_t_i = t_i // 2
                    s2t_qas.append({
                        'id': f'{pair_id}_{tok_id}_{t_i}_s2t_src_gap',
                        'question': question,
                        'is_impossible': False,
                        'answers': [get_answer_info(mt_line, mapped_t_i, mapped_t_i), ]
                    })
                    possible_count += 1
            else:
                if args.no_aligned_gap_source_word_policy == 'skip':
                    pass
                elif args.no_aligned_gap_source_word_policy == 'empty':
                    s2t_qas.append(empty_record)
                    impossible_count += 1
                elif args.no_aligned_gap_source_word_policy == 'align_to_word':
                    spans = point2span(src_mt_align[tok_id])
                    for span_id, span in enumerate(spans):
                        start_i, end_i = min(span), max(span)
                        s2t_qas.append({
                            'id': f'{pair_id}_{tok_id}_{span_id}_s2t_src_mt',
                            'question': question,
                            'is_impossible': False,
                            'answers': [get_answer_info(mt_line, start_i, end_i)]
                        })
                        possible_count += 1
        else:
            s2t_qas.append(empty_record)
            impossible_count += 1

    # collect t2s qas
    t2s_qas = []
    t2s_res = {
        'paragraphs': [{
            'context': src_line,
            'qas': t2s_qas
        }],
        'title': f'{pair_id}_t2s'
    }
    for tok_id, token in enumerate(mt_tokens):
        question = generate_quesiton_str(mt_tokens, tok_id)
        empty_record = {
            'id': f'{pair_id}_{tok_id}_t2s',
            'question': question,
            'is_impossible': True,
            'answers': [{'answer_start': -1, 'text': ''}]
        }
        if gap_src_align is not None:
            # generating training data
            mapped_tok_id = tok_id * 2
            if len(gap_src_align[mapped_tok_id]) > 0:
                for s_i in gap_src_align[mapped_tok_id]:
                    t2s_qas.append({
                        'id': f'{pair_id}_{mapped_tok_id}_{s_i}_t2s_gap_src',
                        'question': question,
                        'is_impossible': False,
                        'answers': [get_answer_info(src_line, s_i, s_i), ]
                    })
                    possible_count += 1
            else:
                if args.no_aligned_gap_source_word_policy == 'skip':
                    pass
                elif args.no_aligned_gap_source_word_policy == 'empty':
                    t2s_qas.append(empty_record)
                    impossible_count += 1
                elif args.no_aligned_gap_source_word_policy == 'align_to_word':
                    spans = point2span(mt_src_align[tok_id])
                    for span_id, span in enumerate(spans):
                        start_i, end_i = min(span), max(span)
                        t2s_qas.append({
                            'id': f'{pair_id}_{tok_id}_{span_id}_t2s_mt_src',
                            'question': question,
                            'is_impossible': False,
                            'answers': [get_answer_info(src_line, start_i, end_i)]
                        })
                        possible_count += 1

        else:
            # generating testing data
            t2s_qas.append(empty_record)
            impossible_count += 1

    mt_len = len(mt_tokens)
    if gap_src_align and len(gap_src_align[mt_len * 2]) > 0:
        for s_i in gap_src_align[mt_len * 2]:
            t2s_qas.append({
                'id': f'{pair_id}_{mt_len}_{s_i}_t2s_gap_src',
                'question': generate_quesiton_str(mt_tokens, mt_len),
                'is_impossible': False,
                'answers': [get_answer_info(src_line, s_i, s_i), ]
            })
            possible_count += 1

    return s2t_res, t2s_res


def process(args):
    # Read in lines
    def read_fn(fn):
        with open(fn) as f:
            return [l.strip() for l in f]

    src_lines = read_fn(args.src)
    mt_lines = read_fn(args.mt)
    std_len = len(src_lines)
    assert len(mt_lines) == std_len, 'Unmatched line number.'

    if args.src_gap_align:
        src_gap_align_lines = read_fn(args.src_gap_align)
        src_mt_align_lines = read_fn(args.src_mt_align) if args.src_mt_align else [None for _ in range(std_len)]
    else:
        src_gap_align_lines = src_mt_align_lines = [None for _ in range(std_len)]

    assert len(src_gap_align_lines) == len(src_mt_align_lines) == std_len, 'Unmatched line number.'

    src_gap_aligns = parse_alignment(src_gap_align_lines)
    src_mt_aligns = parse_alignment(src_mt_align_lines)

    # process one pair
    data_res = []
    pair_id = 0
    global possible_count
    global impossible_count
    possible_count, impossible_count = 0, 0
    for src_line, mt_line, src_gap_align, src_mt_align in zip(src_lines, mt_lines, src_gap_aligns, src_mt_aligns):
        s2t_res, t2s_res = process_one_pair(pair_id, src_line, mt_line, src_gap_align, src_mt_align, args)
        data_res.append(s2t_res)
        data_res.append(t2s_res)
        pair_id += 1

    return possible_count, impossible_count, data_res


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', help='Path to the source file.')
    parser.add_argument('-t', '--mt', help='Path to the MT file WITHOUT ANY GAP TOKENS.')
    parser.add_argument('-o', '--output', help='Path to the output json file.')

    parser.add_argument('-a', '--src_gap_align', default=None,
                        help='Path to the SOURCE-GAP alignment file.'
                             'NOTE: INDEX OF GAP IS THAT IN COMPLETE MT SENTENCE.')
    parser.add_argument('--no_aligned_gap_source_word_policy', choices=['skip', 'empty', 'align_to_word'],
                        default='align_to_word',
                        help='Choose a mode determining whether adding a record of a source word without any aligned '
                             'gaps into the output. Note that it also affects the default mode for the t2s direciton.'
                             'Only valid when -a is specified generating training data. Default: skip.')
    parser.add_argument('--src_mt_align', default=None,
                        help='Path to the source-mt alignment. Required if no_aligned_gap_source_word_policy is set '
                             'to src_mt.')

    # parser.add_argument('--gap_token', default='[GAP]', help='the token representing gaps. Default: [GAP]')
    parser.add_argument('--special_token', default='¶', help='a special token marking out corresponding word. '
                                                             'Default: ¶')

    args = parser.parse_args()

    if args.src_gap_align is None:
        assert args.src_mt_align is None, \
            'Are you generating test data since you did not set -a? If so, please do not set --src_mt_align since ' \
            'it\'s meanless.'
    elif args.no_aligned_gap_source_word_policy == 'align_to_word':
        assert args.src_mt_align is not None, \
            'You should specify SRC-MT alignment when no_aligned_gap_source_word_policy is align_to_word.'

    return args


def main():
    args = parse_args()
    possible_count, impossible_count, data = process(args)
    content = {
        'version': 'v2.0',
        'data': data
    }
    with open(args.output, 'w') as wf:
        wf.write(json.dumps(content, indent=4))

    print(f'{possible_count} answerable records and {impossible_count} unanswerable records are written')


if __name__ == '__main__':
    main()
