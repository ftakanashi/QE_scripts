#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import math
import random
from scipy.stats import norm

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ref',
                        help='Path to the reference tag file.')
    parser.add_argument('-h1', '--hyp1',
                        help='Path to the first hypothesis tag file.')
    parser.add_argument('-h2', '--hyp2',
                        help='Path to the second hypothesis tag file.')
    parser.add_argument('-n', '--number',
                        help='Number of samples extracted to do the test.')
    parser.add_argument('--p_value', type=float, default=0.05,
                        help='P-value for test. Default: 0.05')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    def read_f(fn):
        with open(fn, 'r') as f:
            return [l.strip() for l in f]

    ref_lines = read_f(args.ref)
    hyp1_lines = read_f(args.hyp1)
    hyp2_lines = read_f(args.hyp2)

    # flatten tags
    ref_tags, hyp1_tags, hyp2_tags = [], [], []
    for ref, hyp1, hyp2 in zip(ref_lines, hyp1_lines, hyp2_lines):
        ref_toks, hyp1_toks, hyp2_toks = ref.split(), hyp1.split(), hyp2.split()
        assert len(ref_toks) == len(hyp1_toks) == len(hyp2_toks)
        ref_tags.extend(ref_toks)
        hyp1_tags.extend(hyp1_toks)
        hyp2_tags.extend(hyp2_toks)

    # sampling
    N = len(ref_tags)
    sample_indices = random.sample(range(N), args.number)
    ref_sample = [ref_tags[i] for i in sample_indices]
    hyp1_sample = [hyp1_tags[i] for i in sample_indices]
    hyp2_sample = [hyp2_tags[i] for i in sample_indices]

    # sigificance test
    n1 = n2 = args.number
    x1 = x2 = 0
    for r, h in zip(ref_sample, hyp1_sample):
        if r == h: x1 += 1
    for r, h in zip(ref_sample, hyp2_sample):
        if r == h: x2 += 1

    print(f'{args.hyp1}: ({x1}/{n1})')
    print(f'{args.hyp2}: ({x2}/{n2})')
    p1 = x1 / float(n1)
    p2 = x2 / float(n2)
    p = (x1 + x2) / float((n1 + n2))
    z = abs(p1 - p2) / math.sqrt(p * (1-p) * (1.0/n1 + 1.0/n2))
    p_val_z = norm.ppf(1 - args.p_value)

    print(f'Z = {z}')
    print(f'Z @p-value = {p_val_z}')
    if z >= p_val_z:
        print('Significant Different.')
    else:
        print('NOT Significant Different.')

if __name__ == '__main__':
    main()