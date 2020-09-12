#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections
import json
import os

from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-np', '--nbest_predictions',
                        help='Path to the nbest prediction file.')
    parser.add_argument('-od', '--output-dir',
                        help='A directory to save all the output data.')

    args = parser.parse_args()

    return args


def process_sent_infos(sent_infos, args):
    for sent_id, sent_info in sent_infos.items():
        wf = Path(os.path.join(args.output_dir, f'sent_{sent_id}.lp')).open('w')

        def printw(msg, **kwargs):
            print(msg, file=wf, **kwargs)

        # writing methods and objects
        printw('Minimize')
        printw('obj:')

        def _gen_direc_val(direc):
            for var, prob in sent_info[direc].items():
                direc, word_id, s, e = var.split('_')
                if s == 'X' or e == 'X':
                    val = prob
                else:
                    val = 1 - prob

                printw(f'+ {val} x{var}')

        _gen_direc_val('s2t')
        _gen_direc_val('t2s')

        # writing subjects
        printw('\nSubject to\n')

        subjects = collections.defaultdict(list)
        for var in sent_info['s2t']:
            direc, word_id, s, e = var.split('_')
            subjects[f'{direc}_{word_id}'].append(var)
        for var in sent_info['t2s']:
            direc, word_id, s, e = var.split('_')
            subjects[f'{direc}_{word_id}'].append(var)

        for subject_title, subject_items in subjects.items():
            printw(f'{subject_title}:')
            for subject_item in subject_items:
                printw(f'+ 1 x{subject_item}')
            printw('= 1\n')

        # writing Binary declaration
        printw('Binary')
        for var in sent_info['s2t']:
            printw(f'x{var}')
        for var in sent_info['t2s']:
            printw(f'x{var}')

        printw('End')


def main():
    args = parse_args()

    with Path(args.nbest_predictions).open() as f:
        data = json.load(f)

    sent_infos = {}
    for q_id, answers in data.items():
        sent_id, word_id, direc = q_id.split('_')
        sent_id, word_id = map(int, (sent_id, word_id))
        if sent_id not in sent_infos:
            sent_info = {'s2t': collections.OrderedDict(), 't2s': collections.OrderedDict()}
            sent_infos[sent_id] = sent_info
        else:
            sent_info = sent_infos[sent_id]

        for a in answers:
            s = a['start_index'] if a['start_index'] >= 0 else 'X'
            e = a['end_index'] if a['end_index'] >= 0 else 'X'
            sent_info[direc][f'{direc}_{word_id}_{s}_{e}'] = a['probability']

    process_sent_infos(sent_infos, args)


if __name__ == '__main__':
    main()
