#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
''' 
    This script takes source/target corpus (optionally, alignment and source/target tags as well) as input.
    A SQuAD v2.0 format JSON file will be generated based on the provided files. Such a file is used to fine-tune a 
    *BERT model to do word alignment mimicking QA task in BERT settings.
    
    If -a is not set, answers in JSON will be empty.
    If tags are not set, tag field (if exists, controlled by the flag --with_tags) will be empty.
    
    Usage:
    
    (building train_file) python build_squad_with_tag.py 
    -s train.src -t train.mt -o src_mt_train.for_train.json 
    -a src_mt_train.fast_align.align --with_tags --src_tags train.source_tags --tgt_tags train.tags
    
    (building predict_file) python build_squad_with_tag.py 
    -s dev.src -t dev.mt -o src_mt_dev.for_pred.json 
    --with_tags
'''

import argparse
import collections
import copy
import json
import random
import warnings

from pathlib import Path


def analyze_src_gap_alignment(align_lines):
    '''
    By default, we expect indices of MT gaps to be the one in full-width MT.
    Namely, 0 is the first gap, 2 is the second, 4 is the third, and so on.
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


def calc_answer_start(tokens, span, src2gap):
    '''
    calculate the start index of answer in character unit.
    '''
    start_tok_i = min(span)
    if start_tok_i > 0 and src2gap:
        start_tok_i = start_tok_i // 2 - 1
    total_len = start_tok_i  # length of spaces
    for t in tokens[:start_tok_i]:
        total_len += len(t)
    return total_len


def point2span(points, src2gap=False):
    '''
    detect all spans in a list of non-continuous indices
    E.g. [1,2,4,5,6,8]  ->  [[1,2], [4,5,6], [8,]]
    '''
    if src2gap:
        return [[n,] for n in points]
    sorted_span = list(sorted(points))
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


def make_aug_front(tokens, query_i, spec_tok, src2gap):
    tmp = copy.copy(tokens)
    if src2gap:
        tmp.insert(query_i, spec_tok)
        tmp.insert(query_i + 2, spec_tok)
    else:
        tmp.insert(query_i, f"{spec_tok} {spec_tok}")
    return ' '.join(tmp)


def process_one_pair(src_line, tgt_line, s2t_align, t2s_align, pair_id, args):
    '''
    Process one pair of lines, namely one src_line and one tgt_line.
    Among one pair, a source-to-target and a reverse target-to-source context-qas group will be generated.
    A context-qa group means
    {
        'context': 'xxx',
        'qas': [
            {'id': xxx, 'question': 'xxx', 'tag': 'xxx', answers: [{xxx},{xxx}], is_impossible: bool},
            ...
        ]
    }
    Returning: SRC2TGT_GROUP, TGT2SRC_GROUP, number of possible qa, number of impossible qas
    '''

    src_tokens = src_line.split()
    tgt_tokens = tgt_line.split()

    possible_count = impossible_count = 0

    def process_one_direction(from_tokens, to_tokens, align, res_container, pair_id, id_flag,
                              null_dropout, src2gap=True):
        '''
        Process one direction.
        For every token in from_tokens, build up a QA and append it to the res_container. Whether the QA has an
        answer depends on the alignment.
        '''

        pos_count = imp_count = 0  # partial counts within the direction under processing
        iterator = range(len(from_tokens)) if src2gap else range(len(from_tokens) + 1)
        for i in iterator:
            # insert special tokens to mark the token being processed
            aug = make_aug_front(from_tokens, i, args.special_token, src2gap=src2gap)
            to_span = align[i] if src2gap else align[i * 2]  # a list of aligned target indexs. May be empty list.

            if len(to_span) == 0:  # there is no answer

                # randomly drop impossible cases
                if random.random() < null_dropout: continue

                is_impossible = True
                imp_count += 1

                answers = [{'answer_start': -1, 'text': ''}]
                res = {
                    'id': f'{pair_id}_{i}_{id_flag}' if id_flag is not None else f'{pair_id}_{i}',
                    'question': aug,
                    'answers': answers,
                    'is_impossible': is_impossible
                }

                res_container.append(res)

            else:  # at least one answer
                is_impossible = False
                pos_count += 1
                spans = point2span(to_span, src2gap=src2gap)  # cut the non-continuous span into pieces

                # multiple answers will generate multiple QAs
                for span_id, span in enumerate(spans):
                    answers = [{
                        'answer_start': calc_answer_start(to_tokens, span, src2gap),
                    }]
                    if src2gap:
                        start_tok_i = span[0] // 2 - 1
                        if start_tok_i < 0:
                            answers[0]["text"] = to_tokens[0]
                        elif start_tok_i == len(to_tokens) - 1:
                            answers[0]["text"] = to_tokens[-1]
                        else:
                            answers[0]["text"] = to_tokens[start_tok_i] + " " + to_tokens[start_tok_i + 1]
                    else:
                        answers[0]["text"] = " ".join([to_tokens[t] for t in span])
                    res = {
                        'id': f'{pair_id}_{i}_{span_id}_{id_flag}' if id_flag is not None else \
                            f'{pair_id}_{i}_{span_id}',
                        'question': aug,
                        'answers': answers,
                        'is_impossible': is_impossible
                    }

                    res_container.append(res)

        return pos_count, imp_count

    s2t_qas = []
    p_c, imp_c = process_one_direction(src_tokens, tgt_tokens, s2t_align, s2t_qas, pair_id, 's2t',
                                       args.null_dropout, src2gap=True)
    possible_count += p_c
    impossible_count += imp_c

    t2s_qas = []
    p_c, imp_c = process_one_direction(tgt_tokens, src_tokens, t2s_align, t2s_qas, pair_id, 't2s',
                                       args.null_dropout, src2gap=False)
    possible_count += p_c
    impossible_count += imp_c

    s2t_context = ' '.join(tgt_line.strip().split())
    t2s_context = ' '.join(src_line.strip().split())

    return {
               'context': s2t_context,
               'qas': s2t_qas
           }, {
               'context': t2s_context,
               'qas': t2s_qas
           }, possible_count, impossible_count


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


def main():
    args = parse_args()

    def read_fn(fn):
        with Path(fn).open(encoding='utf-8') as f:
            return [l.strip() for l in f]

    src_lines = read_fn(args.src)
    tgt_lines = read_fn(args.tgt)

    std_len = len(src_lines)
    assert len(tgt_lines) == std_len, 'Number of lines does not match for source and target corpus.'

    s2t_align_lines = t2s_align_lines = ['' for _ in range(std_len)]
    if args.align is not None:
        s2t_align_lines = read_fn(args.align)
        t2s_align_lines = reverse_align_lines(s2t_align_lines)
        assert len(s2t_align_lines) == std_len, \
            f'Number of lines does not match. {std_len} {len(s2t_align_lines)}'

    s2t_aligns = analyze_src_gap_alignment(s2t_align_lines)
    t2s_aligns = analyze_src_gap_alignment(t2s_align_lines)

    possible_count = impossible_count = 0  # counter will be printed in stdout for notice

    data = []  # main container
    for i, src_line in enumerate(src_lines):
        tgt_line = tgt_lines[i]

        # following could be empty string
        s2t_align = s2t_aligns[i]
        t2s_align = t2s_aligns[i]

        if len(s2t_align) == 0 and len(t2s_align) == 0:
            continue

        s2t_one_pair_res_para, t2s_one_pair_res_para, p_c, imp_c = \
            process_one_pair(src_line, tgt_line, s2t_align, t2s_align, i, args)

        s2t_data = {'paragraphs': [s2t_one_pair_res_para, ], 'title': f'{i}_s2t'}
        t2s_data = {'paragraphs': [t2s_one_pair_res_para, ], 'title': f'{i}_t2s'}

        data.append(s2t_data)
        data.append(t2s_data)

        possible_count += p_c
        impossible_count += imp_c

    # when the corpus is huge, number of examples in squad json tends to be very huge.
    # in some toolkit (like google-research/bert), all examples are dumped to memory once.
    # in case that memory won't be consumed totally, you could split it into several shards
    shard_len = len(data) // args.shards
    for shard_i in range(args.shards):
        data_split = data[shard_i * shard_len:(shard_i + 1) * shard_len]

        res = {
            'version': 'v2.0',
            'data': data_split
        }
        content = json.dumps(res, indent=2, ensure_ascii=False)
        if args.shards == 1:
            fn = args.output
        else:
            fn = f'{args.output}.{shard_i}'
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'shard[{shard_i}] written')

    print(f'Extracted {possible_count} possible qas and {impossible_count} impossible qas.')


def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-s', '--src', required=True)
    parser.add_argument('-t', '--tgt', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-a', '--align', default=None,
                        help='If set to None, it means that the data might be test data without golden alignments.')

    parser.add_argument('--null_dropout', type=float, default=0.0,
                        help='In order to adjust and get a proper null ratio, simply randomly ignore some null answer'
                             ' qa cases.')

    parser.add_argument('--shards', type=int, default=1,
                        help='When processing huge dataset, split the json into several shards.')

    parser.add_argument('--special_token', default='¶',
                        help='The special token that mark a word in the query. Make sure that the token is in '
                             'vocabulary of mBERT.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
