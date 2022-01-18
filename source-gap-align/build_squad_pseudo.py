#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
import copy
import json
import random
import uuid

from pathlib import Path


def analyze_src_pe_alignment(align_lines):
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
            a, b = map(int, align.split('-'))
            align_info[a].append(b)
        res.append(align_info)

    return res

def calc_answer_start(tokens, start_tok_i):
    '''
    calculate the start index of answer in character unit.
    '''
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

def make_aug_front(tokens, query_i, spec_tok, src2pe):
    tmp = copy.copy(tokens)
    if src2pe:
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

    data = [
        {"title": f"s2t_{pair_id}", "paragraphs": []},
        {"title": f"t2s_{pair_id}", "paragraphs": []}
    ]
    s2t_paragraphs = data[0]["paragraphs"]
    t2s_paragraphs = data[1]["paragraphs"]

    # s2t non-drop samples
    non_drop_sample = {
        "context": " ".join(tgt_line.strip().split()),
        "qas": []
    }
    qas = non_drop_sample["qas"]
    for src_token_i in range(len(src_tokens)):
        if random.random() < args.dropout: continue
        qas.append({
            "id": str(uuid.uuid4()),
            "question": make_aug_front(src_tokens, src_token_i, args.special_token, True),
            "answers": [{"answer_start": -1, "text": ""}],
            "is_impossible": True
        })
        impossible_count += 1
    s2t_paragraphs.append(non_drop_sample)

    # t2s non-drop samples
    non_drop_sample = {
        "context": " ".join(src_line.strip().split()),
        "qas": []
    }
    qas = non_drop_sample["qas"]
    for tgt_token_i in range(len(tgt_tokens) + 1):
        if random.random() < args.dropout: continue
        qas.append({
            "id": str(uuid.uuid4()),
            "question": make_aug_front(tgt_tokens, tgt_token_i, args.special_token, False),
            "answers": [{"answer_start": -1, "text": ""}],
            "is_impossible": True
        })
        impossible_count += 1
    t2s_paragraphs.append(non_drop_sample)

    # s2t drop samples
    s2t_drop_samples = []
    for src_token_i in range(len(src_tokens)):
        tgt_token_is = s2t_align[src_token_i]

        if len(tgt_token_is) == 0: continue
        if random.random() < args.dropout: continue

        tgt_token_spans = point2span(tgt_token_is)

        question = make_aug_front(src_tokens, src_token_i, args.special_token, True)
        for span_i, span in enumerate(tgt_token_spans):
            left, right = min(span), max(span)
            dropped_tgt_tokens = tgt_tokens[:left] + tgt_tokens[right+1:]

            s2t_sample = {"context": " ".join(dropped_tgt_tokens), "qas": []}
            answer_span_l = max(0, left - 1)
            answer_span_r = min(len(dropped_tgt_tokens)-1, left)
            answer_text = " ".join(dropped_tgt_tokens[answer_span_l:answer_span_r+1])

            qa = {
                "id": str(uuid.uuid4()),
                "question": question,
                "answers": [{
                    "answer_start": calc_answer_start(dropped_tgt_tokens, answer_span_l),
                    "text": answer_text
                }],
                "is_impossible": False
            }
            possible_count += 1

            s2t_sample["qas"].append(qa)
            s2t_drop_samples.append(s2t_sample)

    # t2s drop samples
    t2s_drop_samples = [{"context": " ".join(src_tokens), "qas": []}]
    for tgt_token_i in range(len(tgt_tokens)):
        src_token_is = t2s_align[tgt_token_i]
        if len(src_token_is) == 0: continue
        if random.random() < args.dropout: continue

        src_token_spans = point2span(src_token_is)

        question = make_aug_front(tgt_tokens[:tgt_token_i] + tgt_tokens[tgt_token_i + 1:],
                                  tgt_token_i, args.special_token, False)
        for span_i, span in enumerate(src_token_spans):
            answer_start = calc_answer_start(src_tokens, min(span))
            answer_text = " ".join([src_tokens[i] for i in span])
            qa = {
                "id": str(uuid.uuid4()),
                "question": question,
                "answers": [{
                    "answer_start": answer_start,
                    "text": answer_text
                }],
                "is_impossible": False
            }
            possible_count += 1
            t2s_drop_samples[0]["qas"].append(qa)

    s2t_paragraphs.extend(s2t_drop_samples)
    t2s_paragraphs.extend(t2s_drop_samples)

    return data, possible_count, impossible_count

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

    s2t_aligns = analyze_src_pe_alignment(s2t_align_lines)
    t2s_aligns = analyze_src_pe_alignment(t2s_align_lines)

    possible_count = impossible_count = 0  # counter will be printed in stdout for notice

    data = []  # main container
    for i, src_line in enumerate(src_lines):
        tgt_line = tgt_lines[i]
        s2t_align = s2t_aligns[i]
        t2s_align = t2s_aligns[i]

        row_data, p_c, imp_c = process_one_pair(src_line, tgt_line, s2t_align, t2s_align, i, args)

        data.extend(row_data)

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
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', required=True,
                        help="Path to the source corpus.")
    parser.add_argument('-t', '--tgt', required=True,
                        help="Path to the PE corpus.")
    parser.add_argument('-o', '--output', required=True,
                        help="Path to the output file.")
    parser.add_argument('-a', '--align', required=True,
                        help='Path to the source-PE alignment.')

    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Rate of dropping out some cases")

    parser.add_argument('--special_token', default='¶',
                        help='The special token that mark a word in the query. Make sure that the token is in '
                             'vocabulary of mBERT.')

    parser.add_argument('--shards', type=int, default=1,
                        help='When processing huge dataset, split the json into several shards.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
