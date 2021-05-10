#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    Take a generate log produced by generate.py of fairseq as the input.
    Analyze it and output one of the subset from the log, which is either source sentences, target sentences and hypothesis.
    Token-wise probabilities is under development.
'''

import argparse
import re

from tqdm import tqdm

VALID_TYPE2PATTERN = {
    'hyp': '^H\-(\d+?)\t(.+?)\t(.+?)$',
    'src': '^S\-(\d+?)\t(.+?)$',
    'tgt': '^T\-(\d+?)\t(.+?)$',
    # 'prob': '^P\-(\d+?)\t(.+?)$',
}
VALID_TYPES = VALID_TYPE2PATTERN.keys()

class Entry:
    def __init__(self, type, sent_id, text=None):
        assert type in VALID_TYPES, f'Invalid type {type}'
        self.type = type
        self.sent_id = int(sent_id)
        self.text = text

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to the input log file.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')

    parser.add_argument('-t', '--type', default='hyp', choices=VALID_TYPES,
                        help='A symbol indicating what type of sentences to extract. Default: hyp.')
    parser.add_argument('--retain_original_order', action='store_true',
                        help='Set this flag to avoid ordering by sentence id. Default: False.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    res = []
    ptn = VALID_TYPE2PATTERN[args.type]
    input_f = open(args.input, 'r')
    for line in tqdm(input_f, mininterval=1):
        m = re.search(ptn, line)
        if m is None: continue
        entry = Entry(args.type, m.group(1))
        if args.type == 'hyp':
            entry.text = m.group(3).strip()
        else:
            entry.text = m.group(2).strip()
        res.append(entry)

    input_f.close()

    output_f = open(args.output, 'w')
    if not args.retain_original_order:
        print('Resorting...')
        res.sort(key=lambda x:x.sent_id)

    print('Writing output file...')
    for e in res:
        output_f.write(e.text + '\n')
    output_f.close()

if __name__ == '__main__':
    main()