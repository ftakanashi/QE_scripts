#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
    '''
        For word alignment purpose.
        Given a nbest_predictions_.json produced by run_squad_huggingface_bert.py
        Output a .lp file for every sentence pair which will be passed to CPLEX to do optimization by adopting integer 
        linear programming.
        The default subject is that in both s2t or t2s direction, each word can only be assigned to an aligned word or 
        null(which means aligns to no word) ONCE.
    '''

import argparse
import collections
import json
import os
import re
import subprocess

from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-np', '--nbest_predictions',
                        help='Path to the nbest prediction file.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')
    parser.add_argument('--tmp-working-dir', default='tmp',
                        help='A working dir for temporarily saving .lp files and etc.'
                             'DEFAULT: tmp')
    parser.add_argument('--cplex-bin', default='/opt/ibm/cplex',
                        help='Path to the CPLEX binary file. DEFAULT: /opt/ibm/cplex')

    args = parser.parse_args()

    return args


def generate_sent_lps(sent_infos, args):
    if not os.path.exists(args.tmp_working_dir):
        os.makedirs(args.tmp_working_dir)
    elif not os.path.isdir(args.tmp_working_dir):
        raise ValueError(f'tmp working directory {args.tmp_working_dir} is not a valid dir.')

    for sent_id, sent_info in tqdm(sent_infos.items(), mininterval=0.5, ncols=100, desc='generating .lp for each '
                                                                                        'sentence pair'):
        wf = Path(os.path.join(args.tmp_working_dir, f'sent_{sent_id}.lp')).open('w')

        def printw(msg, **kwargs):
            print(msg, file=wf, **kwargs)

        # writing methods and objects
        printw('Minimize')
        printw('obj:')

        def _gen_direc_val(direc):
            for var, prob in sent_info[direc].items():
                direc, word_id, s, e = var.split('_')
                if s == 'X' or e == 'X':
                    assert s == 'X' and e == 'X', 'start index and end index are not both -1.'
                    val = 1 - prob
                else:
                    val = 1 - prob
                    s, e = map(int, (s, e))
                    val *= (e - s + 2) / 2

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


def cplex_stdout_analyze(output):
    for l in output.decode('utf-8').split('\n'):
        m = re.match('^x(.+?)\s+1\.0+$', l.strip())
        if m is not None:
            yield m.group(1)


def optimize_and_analyze(sent_infos, args):
    # cplex_bin = '/nfs/gshare/optimizer_nishino/CPLEX_Studio128/cplex/bin/x86-64_linux/cplex'
    cplex_bin = args.cplex_bin
    tmp_dir = args.tmp_working_dir

    def run_cmd(cmd):
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        if err:
            raise RuntimeError(f'Error Occured When Running CMD [{cmd}]:\n{err}')
        else:
            return out

    align_lines = []
    for sent_id in tqdm(range(len(sent_infos)), mininterval=0.5, ncols=100, desc='Optimizing and fetching align '
                                                                                 'results'):
        lp_fn = os.path.join(tmp_dir, f'sent_{sent_id}.lp')
        assert os.path.isfile(lp_fn), f'{lp_fn} is not generated yet.'
        output = run_cmd(f'{cplex_bin} -c "read {lp_fn}" "optimize" "display solution variables x*"')
        aligns = []
        for a in cplex_stdout_analyze(output):
            direc, word_id, s, e = a.split('_')
            if s == 'X': s = -1
            if e == 'X': e = -1
            word_id, s, e = map(int, (word_id, s, e))

            if s == -1 or e == -1:
                continue

            if direc == 's2t':
                aligns.extend([(word_id, j) for j in range(s, e + 1)])
            elif direc == 't2s':
                aligns.extend([(i, word_id) for i in range(s, e + 1)])
            else:
                raise ValueError(f'Invalid direction {direc}')

        aligns = list(set(aligns))
        align_lines.append(sorted(aligns))

    return align_lines


def main():
    args = parse_args()

    with Path(args.nbest_predictions).open() as f:
        data = json.load(f)

    sent_infos = {}
    for q_id, answers in data.items():
        sent_id, word_id, direc = q_id.split('_')
        sent_id, word_id = map(int, (sent_id, word_id))

        try:
            sent_info = sent_infos[sent_id]
        except KeyError as e:
            sent_info = {'s2t': collections.OrderedDict(), 't2s': collections.OrderedDict()}
            sent_infos[sent_id] = sent_info

        for a in answers:
            s = a['start_index'] if a['start_index'] >= 0 else 'X'
            e = a['end_index'] if a['end_index'] >= 0 else 'X'
            sent_info[direc][f'{direc}_{word_id}_{s}_{e}'] = a['probability']

    generate_sent_lps(sent_infos, args)

    align_lines = optimize_and_analyze(sent_infos, args)

    wf = Path(args.output).open('w')
    for aligns in align_lines:
        wf.write(' '.join([f'{i}-{j}' for i, j in aligns]) + '\n')
    wf.close()


if __name__ == '__main__':
    main()
