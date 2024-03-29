#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
''' 
This script takes source/target corpus (optionally, alignment and source/target tags as well) as input.
A SQuAD v2.0 format JSON file will be generated based on the provided files. Such a file is used to fine-tune a 
*BERT model to do word alignment mimicking QA task in BERT settings.

If -a is not set, answers in JSON will be empty.
If tags are not set, tag field (if exists, controlled by the flag --with_tags) will be empty.

Example of usage:
(building train_file)
python build_squad_with_tag.py 
-s train.src -t train.mt -o src_mt_train.for_train.json 
-a src_mt_train.fast_align.align --with_tags --src_tags train.source_tags --tgt_tags train.tags

(building predict_file)
python build_squad_with_tag.py -s dev.src -t dev.mt -o src_mt_dev.for_pred.json --with_tags
'''

import argparse
import collections
import copy
import json
import random
import warnings

from pathlib import Path


def analyze_alignment(align_lines, args):
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
            if args.alignment_start_from_one:
                align_info[s - 1].append(t - 1)
            else:
                align_info[s].append(t)
        res.append(align_info)

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


def point2span(span):
    '''
    detect all spans in a list of non-continuous indices
    E.g. [1,2,4,5,6,8]  ->  [[1,2], [4,5,6], [8,]]
    '''
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


def make_aug_src(src_tokens, query_i, spec_tok):
    tmp = copy.copy(src_tokens)
    tmp.insert(query_i, spec_tok)
    tmp.insert(query_i + 2, spec_tok)
    return ' '.join(tmp)


def process_one_pair(src_line, src_tags_line,
                     tgt_line, tgt_tags_line,
                     s2t_align, t2s_align,
                     pair_id, args):
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

    if src_tags_line != '':
        src_tags = src_tags_line.split()
        assert len(src_tags) == len(src_tokens), f'Number of tokens and tags does not match: {len(src_tokens)} ' \
                                                 f'{len(src_tags)}'
    else:
        src_tags = []

    if tgt_tags_line != '':
        tgt_tags = tgt_tags_line.split()
        if len(tgt_tags) == 2 * len(tgt_tokens) + 1:  # Gap tags are included in target QE tags
            tgt_tags = tgt_tags[1::2]
        assert len(tgt_tags) == len(tgt_tokens), f'Number of tokens and tags does not match: {len(tgt_tokens)} ' \
                                                 f'{len(tgt_tags)}'
    else:
        tgt_tags = []

    possible_count = impossible_count = 0

    def process_one_direction(from_tokens, from_tags, to_tokens, align, res_container, pair_id, id_flag,
                              null_dropout):
        '''
        Process one direction.
        For every token in from_tokens, build up a QA and append it to the res_container. Whether the QA has an
        answer depends on the alignment.
        '''

        pos_count = imp_count = 0  # partial counts within the direction under processing
        for i in range(len(from_tokens)):
            # insert special tokens to mark the token being processed
            aug = make_aug_src(from_tokens, i, args.special_token)
            to_span = align[i]  # a list of aligned target indexs. May be empty list.
            from_token_tag = from_tags[i] if len(from_tags) > 0 else 'None'

            if len(to_span) == 0:  # there is no answer

                # randomly drop impossible cases
                if random.random() < null_dropout:
                    continue

                is_impossible = True
                imp_count += 1

                answers = [{'answer_start': -1, 'text': ''}]
                res = {
                    'id': f'{pair_id}_{i}_{id_flag}' if id_flag is not None else f'{pair_id}_{i}',
                    'question': aug,
                    'answers': answers,
                    'is_impossible': is_impossible
                }
                if args.with_tags:
                    res['tag'] = from_token_tag

                res_container.append(res)

            else:  # at least one answer
                is_impossible = False
                pos_count += 1
                spans = point2span(to_span)  # cut the non-continuous span into pieces

                # multiple answers will generate multiple QAs
                for span_id, span in enumerate(spans):
                    answers = [{
                        'answer_start': calc_answer_start(to_tokens, span),
                        'text': ' '.join([to_tokens[t] for t in span])
                    }]
                    res = {
                        'id': f'{pair_id}_{i}_{span_id}_{id_flag}' if id_flag is not None else \
                            f'{pair_id}_{i}_{span_id}',
                        'question': aug,
                        'answers': answers,
                        'is_impossible': is_impossible
                    }
                    if args.with_tags:
                        res['tag'] = from_token_tag

                    res_container.append(res)

        return pos_count, imp_count

    s2t_qas = []
    p_c, imp_c = process_one_direction(src_tokens, src_tags, tgt_tokens, s2t_align, s2t_qas, pair_id, 's2t',
                                       args.null_dropout)
    possible_count += p_c
    impossible_count += imp_c

    t2s_qas = []
    p_c, imp_c = process_one_direction(tgt_tokens, tgt_tags, src_tokens, t2s_align, t2s_qas, pair_id, 't2s',
                                       args.null_dropout)
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


    src_tags_lines = tgt_tags_lines = ['' for _ in range(std_len)]
    if args.with_tags:
        if args.src_tags and args.tgt_tags:
            src_tags_lines = read_fn(args.src_tags)
            tgt_tags_lines = read_fn(args.tgt_tags)
            assert len(src_tags_lines) == len(tgt_tags_lines) == std_len, \
                f'Number of lines does not match. {std_len} {len(src_tags_lines)} {len(tgt_tags_lines)}'
    else:
        if args.src_tags or args.tgt_tags:
            warnings.warn('You specified src_tags or tgt_tags without adding flag --with_tags. So the information of '
                          'tags won\'t be included.')


    s2t_aligns = analyze_alignment(s2t_align_lines, args)
    t2s_aligns = analyze_alignment(t2s_align_lines, args)

    possible_count = impossible_count = 0  # counter will be printed in stdout for notice

    data = []  # main container
    for i, src_line in enumerate(src_lines):
        tgt_line = tgt_lines[i]

        # following could be empty string
        s2t_align = s2t_aligns[i]
        t2s_align = t2s_aligns[i]
        src_tags_line = src_tags_lines[i]
        tgt_tags_line = tgt_tags_lines[i]

        s2t_one_pair_res_para, t2s_one_pair_res_para, p_c, imp_c = \
            process_one_pair(src_line, src_tags_line, tgt_line, tgt_tags_line, s2t_align, t2s_align, i, args)

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
        content = json.dumps(res, indent=4)
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
                        help='If set to None, it means that the data might be test data without golden tags.')

    parser.add_argument('--with_tags', action='store_true', default=False,
                        help='Set the flag if you want to add tag information in the output JSON.')
    parser.add_argument('--src_tags', default=None,
                        help='Path to the source tags.')
    parser.add_argument('--tgt_tags', default=None,
                        help='Path to the target tags. If the tags are OK/BAD in QE settings, you could use the '
                             'provided one in which GAP tags are included. The script will automatically ignore '
                             'those tags and only use the word tags.')

    parser.add_argument('--alignment_start_from_one', action='store_true', default=False,
                        help='If the alignment file is presented in the format in which the first token is expressed '
                             'as 1 rather than 0, add this option.')

    parser.add_argument('--null_dropout', type=float, default=0.0,
                        help='In order to adjust and get a proper null ratio, simply randomly ignore some null answer'
                             ' qa cases.')

    parser.add_argument('--shards', type=int, default=1,
                        help='When processing huge dataset, split the json into several shards.')

    parser.add_argument('--special_token', default='¶',
                        help='The special token that mark a word in the query. Make sure that the token is in '
                             'vocabulary of mBERT.')

    args = parser.parse_args()

    assert not (args.src_tags is not None) ^ (args.tgt_tags is not None), \
        'keep source and target tags both set or unset for symmetric reason.'

    return args


if __name__ == '__main__':
    main()
