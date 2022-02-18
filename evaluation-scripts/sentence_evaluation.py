#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
A script used to evalueate HTER score's prediction.
'''

import argparse
import numpy as np

from pathlib import Path
from scipy.stats import pearsonr

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref', type=Path,
                        help='Path to the reference HTER scores.')
    parser.add_argument('--pred', type=Path,
                        help='Path to the prediction HTER scores.')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    def read_hter_scores(fn):
        with fn.open() as f:
            return np.array([float(l.strip()) for l in f])

    ref_scores = read_hter_scores(args.ref)
    pred_scores = read_hter_scores(args.pred)

    assert ref_scores.shape == pred_scores.shape, 'Reference scores and prediction scores does not match in size.'

    pearson = pearsonr(ref_scores, pred_scores)[0]
    diff = ref_scores - pred_scores
    mae = np.abs(diff).mean()
    rmse = (diff ** 2).mean() ** 0.5

    print(f'pearson: {pearson}')
    print(f'mae: {mae}')
    print(f'rmse: {rmse}')

if __name__ == '__main__':
    main()