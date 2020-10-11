#!/usr/bin/env python
# -*- coding:utf-8 -*-

NOTE = \
'''
    Merge the source-pe alignments with pe-mt alignments.
    Generate a source-mt alignment file.
'''

import argparse
import collections
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(NOTE)

    parser.add_argument('-sp', '--source-pe-align', type=Path,
                        help='Path to SRC-PE alignment file.')
    parser.add_argument('-pm', '--pe-mt-align', type=Path,
                        help='Path to PE-MT alignment file.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Path to the output file.')

    parser.add_argument('--possible-delimeters', nargs='+', default=['-', 'p'],
                        help='Options of delimeters used in alignment files. DEFAULT: [-, p]')

    args = parser.parse_args()

    return args

def process_sp_pm_pair(sp_line, pm_line, args):
    sp_aligns = []
    for sp_align in sp_line.split():
        s, p = -1, -1
        for d in args.possible_delimeters:
            if d in sp_align:
                s, p = map(int, sp_align.split(d))
                sp_aligns.append((s, p))
                break
        if s == -1 or p == -1:
            raise ValueError(f'No valid possible delimeters found in {sp_align}.')

    pm_map = collections.defaultdict(list)
    for pm_align in pm_line.split():
        p, m = -1, -1
        for d in args.possible_delimeters:
            if d in pm_align:
                p, m = map(int, pm_align.split(d))
                pm_map[p].append(m)
                break
        if p == -1 or m == -1:
            raise ValueError(f'No valid possible delimeters found in {pm_align}.')

    res = []
    for s, p in sp_aligns:
        for m in pm_map[p]:
            res.append((s, m))

    res = sorted(list(set(res)), key=lambda x:x[0])
    return res


def main():
    args = parse_args()

    with args.source_pe_align.open() as f:
        sp_lines = [l.strip() for l in f]

    with args.pe_mt_align.open() as f:
        pm_lines = [l.strip() for l in f]

    assert len(sp_lines) == len(pm_lines), 'SRC-PE and PE-MT alignment files should contain identical number of lines.'

    sm_lines = []
    for sp, pm in zip(sp_lines, pm_lines):
        sm_aligns = process_sp_pm_pair(sp, pm, args)
        sm_lines.append(' '.join(f'{i}-{j}' for i, j in sm_aligns))

    with args.output.open('w') as wf:
        for l in sm_lines:
            wf.write(l + '\n')


if __name__ == '__main__':
    main()