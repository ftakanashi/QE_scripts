#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    This script is designed for extraction of alignment predictions (and tag predictions if needed) from a 
    predictions_.json file produced by run_squad_align_with_tag.py.
    
    Only when -sto and -tto are given, tag predictions will be extracted. Otherwise, only alignment predictions will 
    be generated.
    
    Usage:
    python pred_to_align_with_tag.py
    -p predictions_.json
    -ao pred.align
    -sto pred.source_tags
    -tto pred.mtword_tags  
'''

import argparse
import collections
import json

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pred-json', default=None,
                        help='Path to predictions_.json')
    parser.add_argument('-ao', '--align-output', required=True,
                        help='Path to the output alignment file.')

    parser.add_argument('-sto', '--src_tags_output', default=None,
                        help='Path to the output Source QE tag file.')
    parser.add_argument('-tto', '--tgt_tags_output', default=None,
                        help='Path to the output Target QE tag file.')

    parser.add_argument('--prob_threshold', type=float, default=0.4)

    args = parser.parse_args()

    assert  not (args.src_tags_output is not None) ^ (args.tgt_tags_output is not None), \
    'If tag information is in predictions and you want to extract it, specify -sto and -tto at the same time.'

    return args


def get_align_info(data):
    '''
    extract information from prediction's json.
    for each sentence pair, a defaultdict with a default empty list will be used as a container to save the information
    about the alignments yield from both directions.
    Specifically, key is 'a-b' while a represents for an token index in source and b for target.
    Value of 'a-b' is a list containing probabilities fecthed from 'a_b_s2t' and 'b_a_t2s' items from the original
    json data.
    Note that the list shall never longer than 2.
    :param data:
    :return:
    '''
    info = {}
    for k in data:
        v = data[k]
        sent_id, tok_id, flag = k.split('_')
        sent_id, tok_id = int(sent_id), int(tok_id)
        if sent_id not in info:
            info[sent_id] = collections.defaultdict(list)
        if v == '' or v[0] == '': continue
        for target_tok_i in range(v[3], v[4] + 1):
            key_name = f'{tok_id}-{target_tok_i}' if flag == 's2t' else f'{target_tok_i}-{tok_id}'
            info[sent_id][key_name].append(v[5])

    return info


def process_one_line(sent_align_info, opt):
    '''
    iterate all possible pairs and only record those pairs with an average over threshold
    :param sent_align_info:
    :param opt:
    :return:
    '''

    if len(sent_align_info) == 0:  # no alignment detected
        return []

    alignments = [(int(k.split('-')[0]), int(k.split('-')[1])) for k in sent_align_info.keys()]
    max_src_length = max([a[0] for a in alignments]) + 1
    max_tgt_length = max([a[1] for a in alignments]) + 1

    valid_aligns = []
    for i in range(max_src_length):
        for j in range(max_tgt_length):
            probs = sent_align_info[f'{i}-{j}']
            assert len(probs) <= 2
            if sum(probs) / 2 > opt.prob_threshold:    # symmetrization
                valid_aligns.append(f'{i}-{j}')

    return valid_aligns


def get_tag_info(data):
    src_info = collections.defaultdict(dict)
    tgt_info = collections.defaultdict(dict)
    for k in data:
        sent_id, word_id, dirc = k.split('_')
        sent_id, word_id = map(int, (sent_id, word_id))
        v = data[k]
        if dirc == 's2t':
            info = src_info
        elif dirc == 't2s':
            info = tgt_info
        else:
            raise ValueError(f'Invalid direction {dirc}')

        info[sent_id][word_id] = v[1 if v[0] == '' else 6]

    return src_info, tgt_info


def main():
    args = parse_args()

    with open(args.pred_json, 'r') as f:
        data = json.load(f)

    # extract alignment information
    align_info = get_align_info(data)
    wf = open(args.align_output, 'w')
    for sent_id in tqdm(sorted(align_info), mininterval=1.0, ncols=50, desc='Extracting alignment information'):
        aligns = process_one_line(align_info[sent_id], args)
        wf.write(' '.join(aligns) + '\n')
    wf.close()

    # extract QE tag information
    if args.src_tags_output and args.tgt_tags_output:
        src_tag_info, tgt_tag_info = get_tag_info(data)
        wf = open(args.src_tags_output, 'w')
        for sent_id in tqdm(sorted(src_tag_info), mininterval=1.0, ncols=50, desc='Extracting Source QE Tag Information'):
            wf.write(' '.join([src_tag_info[sent_id][word_id] for word_id in sorted(src_tag_info[sent_id])]) + '\n')
        wf.close()
        wf = open(args.tgt_tags_output, 'w')
        for sent_id in tqdm(sorted(tgt_tag_info), mininterval=1.0, ncols=50, desc='Extracting Target QE Tags information'):
            wf.write(' '.join([tgt_tag_info[sent_id][word_id] for word_id in sorted(tgt_tag_info[sent_id])]) + '\n')
        wf.close()


if __name__ == '__main__':
    main()
