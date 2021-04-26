#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prediction_file',
                        help='Path to the predictions_.json produced by SQuAD prediction model.')
    parser.add_argument('-to', '--tag_output',
                        help='Path to the gap tag output file. If a gap is not aligned to any source words, '
                             'it would be regarded as OK otherwise BAD.')

    parser.add_argument('-ao', '--alignment_output', default=None,
                        help='Path to the alignment output file containing the alignment information between source '
                             'words and MT gaps. Default: None')
    parser.add_argument('--align_prob_threshold', default=0.5, type=float,
                        help='A probability threshold for extracting alignment. Note that currently, the script only '
                             'support one-direction extraction. Therefore, a reasonable value would be around 0.8~0.9.'
                             'Default: 0.5')

    args = parser.parse_args()
    return args

def parse_key(key):
    sent_id, word_id, direc, mode = key.split('_')
    sent_id = int(sent_id)
    word_id = int(word_id)
    return sent_id, word_id, direc, mode

def main():
    args = parse_args()
    with open(args.prediction_file) as f:
        content = json.load(f)

    tag_lines, align_lines = [], []
    tag_line, align_line = [], []
    prev_sent_id = 0
    for k, v in tqdm(content.items(), ncols=50, mininterval=0.5):
        sent_id, word_id, direc, mode = parse_key(k)
        if sent_id > prev_sent_id:
            tag_lines.append(tag_line)
            align_lines.append(align_line)
            tag_line, align_line = [], []
            prev_sent_id = sent_id

        if v == '':
            tag_line.append('OK')
        elif v[-1] >= args.align_prob_threshold:
            tag_line.append('BAD')
            for src_i in range(v[3], v[4]+1):
                align_line.append(f'{src_i}-{word_id}')
        else:
            tag_line.append('OK')
    tag_lines.append(tag_line)
    align_lines.append(align_line)

    with open(args.tag_output, 'w') as wf:
        for tag_line in tag_lines:
            wf.write(' '.join(tag_line) + '\n')

    if args.alignment_output:
        with open(args.alignment_output, 'w') as wf:
            for align_line in align_lines:
                wf.write(' '.join(align_line) + '\n')

if __name__ == '__main__':
    main()