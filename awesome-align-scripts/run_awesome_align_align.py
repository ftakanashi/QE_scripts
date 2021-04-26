#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prefix',
                        help='Prefix of parallel corpus.')
    parser.add_argument('-s', '--src',
                        help='Suffix of source.')
    parser.add_argument('-t', '--tgt',
                        help='Suffix of target.')
    parser.add_argument('-o', '--output_file',
                        help='Path to the output file.')

    parser.add_argument('--trained_model', default='output',
                        help='Path to the trained model. Default: output')
    parser.add_argument('--retain_original_order', action='store_true', default=False,
                        help='Add this flag if you want to retain the order of alignments originally output by '
                             'awesome-align.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size. Default: 32.')
    parser.add_argument('--awesome_align_dir', default='awesome-align',
                        help='Path to the awesome-align project root. Default: awesome-align')

    args = parser.parse_args()
    return args

def run_cmd(cmd):
    flag = os.system(cmd)
    if flag != 0:
        raise Exception(f'Error encountered running [{cmd}]')

def main():
    args = parse_args()

    basefn = os.path.basename(args.prefix) + f'.{args.src}-{args.tgt}'
    # create the concat file
    cmd = f'paste {args.prefix}.{args.src} {args.prefix}.{args.tgt} | awk -F \'\\t\' \'{{print $1 " ||| " $2}}\' >' \
          f' {basefn}'
    run_cmd(cmd)

    # run awesome-align/run_align.py
    cmd = f'python {args.awesome_align_dir}/run_align.py --output_file {args.output_file} ' \
          f'--model_name_or_path {args.trained_model} --data_file {basefn} --extraction \'softmax\' ' \
          f'--batch_size {args.batch_size}'
    run_cmd(cmd)

    if not args.retain_original_order:

        with open(args.output_file, 'r') as f:
            align_lines = [l.strip() for l in f]

        new_lines = []
        for align_line in align_lines:
            align_pairs = align_line.split()
            aligns = []
            for p in align_pairs:
                a, b = map(int, p.split('-'))
                aligns.append((a, b))
            aligns.sort()
            new_lines.append(' '.join([f'{i}-{j}' for i, j in aligns]))

        wf = open(args.output_file, 'w')
        for l in new_lines:
            wf.write(l + '\n')
        wf.close()

if __name__ == '__main__':
    main()