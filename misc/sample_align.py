#!/usr/bin/env python

import argparse
import os
import random
import warnings

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--files', type=str, nargs='+',
                        help='A list of files that need to be sampled.')
    parser.add_argument('-a', '--align',
                        help='Path to the alignment file. Must be one of the file names in the args.files.')
    parser.add_argument('-o', '--output_dir', default=os.getcwd(),
                        help='Path to the output directory. Default: current working dir.')
    parser.add_argument('--sample_size', type=int, default=300,
                        help='Sample size. Default: 300.')

    parser.add_argument('--ignore_empty_align', action='store_true',
                        help='Set this flag to ignore empty alignment lines while sampling.')
    parser.add_argument('--output_indices', type=str, default=None,
                        help='Specify an output file to record the sampled indices in case you need it.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    assert args.align in args.files, f'Cannot find {args.align} in the listed files.'
    pivot_i = args.files.index(args.align)

    def open_fn(fn):
        with open(fn) as f:
            return [l.strip() for l in f.readlines()]

    files_lines = []
    std_len = -1
    for f in args.files:
        lines = open_fn(f)
        files_lines.append(lines)
        if std_len == -1: std_len = len(lines)
        else:
            assert std_len == len(lines), 'Unmatched number of lines.'

    indices = list(range(std_len))
    random.shuffle(indices)
    res_lines = [[] for _ in range(len(files_lines))]
    pivot_lines = files_lines[pivot_i]
    sampled_count = 0
    sampled_indices = []
    for i in indices:
        if args.ignore_empty_align and pivot_lines[i] == '':continue
        for f_i in range(len(files_lines)): res_lines[f_i].append(files_lines[f_i][i])
        sampled_count += 1
        sampled_indices.append(i)
        if sampled_count == args.sample_size:
            break

    if sampled_count < args.sample_size:
        warnings.warn('Please note that there are\'t enough non-empty lines in the alignment file to be sampled '
                      'regarding the specified sample_size.')

    for i, fn in enumerate(args.files):
        base_fn = os.path.basename(fn)
        wf_fn = os.path.join(args.output_dir, base_fn)
        wf = open(wf_fn, 'w')

        for l in res_lines[i]:
            wf.write(l.strip() + '\n')

        wf.close()

    if args.output_indices is not None:
        wf = open(args.output_indices, 'w')
        wf.write('\t'.join([str(i) for i in sampled_indices]))

if __name__ == '__main__':
    main()