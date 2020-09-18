#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    @author: f.takanashi
    A script modified from build_squad.py
    Main difference is that this script support input of QE tags which contains only OK or BAD tags for each word in 
    source and target corpus.
    Note that in default settings, target(MT) tags contains those corresponding to GAP. So we extract target tags
    [1::2] ones as results.
    
    Every QE tag will be added into the squad body following key "question".
    Value of which is "OK" or "BAD".
    Also note that when generating squad json file for testing purpose, QE tags files may not exist.
    In that case, you dont need to specify it and leave it as the default value None.
    When QE tag files are None, a "qe_tag": null K-V pair will be added into the squad json file.
'''

import argparse
import collections
import copy
import json
import random


def analyze_alignment(align_lines, opt):
    res = []
    for align_line in align_lines:
        align_info = collections.defaultdict(list)
        aligns = align_line.strip().split()
        for align in aligns:
            if 'p' in align:
                if opt.only_sure: continue
                s, t = align.split('p')
            else:
                s, t = align.split('-')
            s, t = int(s), int(t)
            if opt.alignment_start_from_one:
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


def split_span(span):
    '''
    cut off a span whenever the element is non-continuous.
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


def make_aug_src(src_tokens, query_i):
    tmp = copy.copy(src_tokens)
    tmp.insert(query_i, '¶')
    tmp.insert(query_i + 2, '¶')
    return ' '.join(tmp)


def process_one_pair(src_line, src_qe_tags_line, tgt_line,
                     tgt_qe_tags_line, s2t_align, t2s_align, pair_id, opt):

    src_tokens = src_line.split()
    src_qe_tags = src_qe_tags_line.split()
    if src_qe_tags_line != '':
        assert len(src_tokens) == len(src_qe_tags), f'Inconsistent QE tags {src_qe_tags} with corpus {src_tokens}'

    tgt_tokens = tgt_line.split()
    tgt_qe_tags = tgt_qe_tags_line.split()[1::2]
    if tgt_qe_tags_line != '':
        assert len(tgt_tokens) == len(tgt_qe_tags), f'Inconsistent QE tags {tgt_qe_tags} with corpus {tgt_tokens}'

    s2t_context = tgt_line
    t2s_context = src_line

    s2t_qas = []
    t2s_qas = []

    possible_count = impossible_count = 0

    def process_line(from_tokens, from_qe_tags, to_tokens, to_qe_tags, align, res_container, pair_id, id_flag, \
                     null_dropout):
        impossible_count = possible_count = 0
        for i in range(len(from_tokens)):
            aug = make_aug_src(from_tokens, i)  # insert special tokens to pack a source token
            to_span = align[i]  # a list of target tokens ids which are aligned to the source token. May be an empty
            # one.
            from_qe_tag = from_qe_tags[i] if len(from_qe_tags) > 0 else None
            if len(to_span) == 0:
                # if there is no answer
                if random.random() < null_dropout:
                    # randomly drop some null QA cases to control the null ratio at a desirable level
                    continue

                is_impossible = True
                impossible_count += 1
                answers = [{'answer_start': -1, 'text': ''}]
                res_container.append({
                    'id': f'{pair_id}_{i}_{id_flag}' if id_flag is not None else f'{pair_id}_{i}',
                    'question': aug,
                    'qe_tag': from_qe_tag,
                    'answers': answers,
                    'is_impossible': is_impossible
                })

            else:
                is_impossible = False
                possible_count += 1
                spans = split_span(to_span)  # cut the non-continuous span into pieces

                if len(spans) > 1 and opt.split_multiple_answers:
                    # multiple answer will be recorded in different qas
                    for span_id, span in enumerate(spans):
                        answers = [{
                            'answer_start': calc_answer_start(to_tokens, span),
                            'text': ' '.join([to_tokens[t] for t in span])
                        }]
                        res_container.append({
                            'id': f'{pair_id}_{i}_{span_id}_{id_flag}' if id_flag is not None else \
                                f'{pair_id}_{i}_{span_id}',
                            'question': aug,
                            'qe_tag': from_qe_tag,
                            'answers': answers,
                            'is_impossible': is_impossible
                        })
                else:
                    # multiple answer will be written into one qa as a list.
                    answers = [{'answer_start': calc_answer_start(to_tokens, span),
                                'text': ' '.join([to_tokens[t] for t in span])}
                               for span in spans]

                    res_container.append({
                        'id': f'{pair_id}_{i}_{id_flag}' if id_flag is not None else f'{pair_id}_{i}',
                        'question': aug,
                        'qe_tag': from_qe_tag,
                        'answers': answers,
                        'is_impossible': is_impossible
                    })

        return possible_count, impossible_count

    s2t_id_flag = 's2t' if opt.do_t2s else None
    null_dropout = opt.null_dropout
    p_c, imp_c = process_line(src_tokens, src_qe_tags, tgt_tokens, tgt_qe_tags, s2t_align, s2t_qas, pair_id,
                              s2t_id_flag, null_dropout)
    possible_count += p_c
    impossible_count += imp_c
    if opt.do_t2s:
        p_c, imp_c = process_line(tgt_tokens, tgt_qe_tags, src_tokens, src_qe_tags, t2s_align, t2s_qas, pair_id, 't2s',
                                  null_dropout)
        possible_count += p_c
        impossible_count += imp_c

    if opt.do_t2s:
        return {
                   'context': s2t_context,
                   'qas': s2t_qas
               }, {
                   'context': t2s_context,
                   'qas': t2s_qas
               }, possible_count, impossible_count
    else:
        return {
                   'context': s2t_context,
                   'qas': s2t_qas
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


def process(opt):
    with open(opt.src, 'r') as f:
        src_lines = [l.strip() for l in f]

    if opt.src_qe_tags is not None:
        with open(opt.src_qe_tags, 'r') as f:
            src_qe_tags_lines = [l.strip() for l in f]
    else:
        src_qe_tags_lines = ['' for _ in range(len(src_lines))]

    with open(opt.tgt, 'r') as f:
        tgt_lines = [l.strip() for l in f]

    if opt.tgt_qe_tags is not None:
        with open(opt.tgt_qe_tags, 'r') as f:
            tgt_qe_tags_lines = [l.strip() for l in f]
    else:
        tgt_qe_tags_lines = ['' for _ in range(len(tgt_lines))]

    if opt.align is not None:
        with open(opt.align, 'r') as f:
            s2t_align_lines = [l.strip() for l in f]

        t2s_align_lines = reverse_align_lines(s2t_align_lines)
    else:
        s2t_align_lines = ['' for _ in range(len(src_lines))]
        t2s_align_lines = ['' for _ in range(len(tgt_lines))]

    s2t_aligns = analyze_alignment(s2t_align_lines, opt)
    t2s_aligns = analyze_alignment(t2s_align_lines, opt)

    possible_count = impossible_count = 0

    data = []
    for i, src_line in enumerate(src_lines):
        tgt_line = tgt_lines[i]
        s2t_align = s2t_aligns[i]
        src_qe_tags_line = src_qe_tags_lines[i]
        t2s_align = t2s_aligns[i]
        tgt_qe_tags_line = tgt_qe_tags_lines[i]
        if opt.do_t2s:
            s2t_one_pair_res_para, t2s_one_pair_res_para, p_c, imp_c = \
                process_one_pair(src_line, src_qe_tags_line, tgt_line, tgt_qe_tags_line, s2t_align, t2s_align, i, opt)

            s2t_data = {'paragraphs': [s2t_one_pair_res_para, ]}
            t2s_data = {'paragraphs': [t2s_one_pair_res_para, ]}
            if opt.add_title:
                s2t_data['title'] = f'{i}_s2t'
                t2s_data['title'] = f'{i}_t2s'

            # data.append({'paragraphs': [s2t_one_pair_res_para, ]})
            # data.append({'paragraphs': [t2s_one_pair_res_para, ]})
            data.append(s2t_data)
            data.append(t2s_data)
        else:
            assert opt.do_t2s, 'Please add do-t2s option. The following code needs maintainance...'
            one_pair_res_para, p_c, imp_c = process_one_pair(src_line, tgt_line, s2t_align, t2s_align, i, opt)
            data.append({'paragraphs': [one_pair_res_para, ]})

        possible_count += p_c
        impossible_count += imp_c

    shard_len = len(data) // opt.shards
    for shard_i in range(opt.shards):
        data_split = data[shard_i * shard_len:(shard_i + 1) * shard_len] if shard_i != opt.shards - 1 else data[
                                                                                                           shard_i * shard_len:]
        res = {
            'version': 'v2.0',
            'data': data_split
        }

        content = json.dumps(res, indent=4)
        if opt.shards == 1:
            fn = opt.output
        else:
            fn = f'{opt.output}.{shard_i}'
        with open(fn, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f'shard[{shard_i}] written')

    print(f'Extracted {possible_count} possible qas and {impossible_count} impossible qas.')
    with open(f'{opt.output}.readme', 'w') as f:
        f.write(f'Extracted {possible_count} possible qas and {impossible_count} impossible qas.')


def main():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-s', '--src', required=True)
    parser.add_argument('-t', '--tgt', required=True)
    parser.add_argument('-a', '--align', default=None,
                        help='If set to None, it means that the data might be test data without golden tags.')
    parser.add_argument('-o', '--output', required=True)

    parser.add_argument('--src-qe-tags', default=None,
                        help='Path to the source QE tags.')
    parser.add_argument('--tgt-qe-tags', default=None,
                        help='Path to the target QE tags.')

    parser.add_argument('--only-sure', action='store_true', default=False)
    parser.add_argument('--do-t2s', action='store_true', default=False)
    parser.add_argument('--split-multiple-answers', action='store_true', default=False,
                        help='When a token is aligned to multiple spans, it is regarded to have multiple answers. In '
                             'SQuAD format, a question could have multiple answers by adding several elements into '
                             'answers\' list. But train data only requires one particular answer so add this option '
                             'when you are generating json for train set.')
    parser.add_argument('--alignment-start-from-one', action='store_true', default=False,
                        help='If the alignment file is presented in the format in which the first token is expressed '
                             'as 1 rather than 0, add this option.')
    parser.add_argument('--add-title', action='store_true', default=False,
                        help='Add "title" in json file to adapt huggingface\'s transformers module.')

    parser.add_argument('--null-dropout', type=float, default=0.0,
                        help='In order to adjust and get a proper null ratio, simply randomly ignore some null answer'
                             ' qa cases.')

    parser.add_argument('--shards', type=int, default=1,
                        help='When processing huge dataset, split the json into several shards.')

    opt = parser.parse_args()

    process(opt)


if __name__ == '__main__':
    main()
