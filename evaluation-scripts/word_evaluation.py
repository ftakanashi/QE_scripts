#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
import os

from pathlib import Path
from sklearn.metrics import f1_score

def list_of_lists(a_list):
    '''
    check if <a_list> is a list of lists
    '''
    if isinstance(a_list, (list, tuple, np.ndarray)) and \
            len(a_list) > 0 and \
            all([isinstance(l, (list, tuple, np.ndarray)) for l in a_list]):
        return True
    return False

def flatten(lofl):
    '''
    convert list of lists into a flat list
    '''
    if list_of_lists(lofl):
        return [item for sublist in lofl for item in sublist]
    elif type(lofl) == dict:
        return lofl.values()

def compute_scores(true_tags, test_tags):
    flat_true = flatten(true_tags)
    flat_pred = flatten(test_tags)

    f1_all_scores = f1_score(flat_true, flat_pred, average=None, pos_label=None)

    # Matthews correlation coefficient (MCC)
    # true/false positives/negatives
    tp = tn = fp = fn = 0
    for pred_tag, gold_tag in zip(flat_pred, flat_true):
        if pred_tag == 1:
            if pred_tag == gold_tag:
                tp += 1
            else:
                fp += 1
        else:
            if pred_tag == gold_tag:
                tn += 1
            else:
                fn += 1

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / mcc_denominator

    return np.append(f1_all_scores, mcc)


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--ref-dir')
    parser.add_argument('--pred-dir')
    parser.add_argument('--evaluate-merged-mt', action='store_true',
                        help='If this option is added, then MT word tags and gap tags will be evaluated together.')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    source_tag_fn = 'source.tags'
    if not args.evaluate_merged_mt:
        mt_tag_fn = 'mt_word.tags'
        gap_tag_fn = 'mt_gap.tags'
    else:
        mt_tag_fn = 'merged_mt.tags'
        gap_tag_fn = None

    TAG_MAP = {
        'OK': 1,
        'BAD': 0
    }

    def read_tag(fn):
        with Path(fn).open() as f:
            return [[TAG_MAP[t] for t in l.strip().split()] for l in f]

    join = os.path.join


    # Source
    ref_source_tags = read_tag(join(args.ref_dir, source_tag_fn))
    pred_source_tags = read_tag(join(args.pred_dir, source_tag_fn))
    f1_bad_src, f1_good_src, mcc_src = compute_scores(ref_source_tags, pred_source_tags)
    f1_multi_src = f1_bad_src * f1_good_src

    print('src_mcc: {:.4}'.format(mcc_src))
    print("src_f1-bad: {:.4}".format(f1_bad_src))
    print('src_f1-good: {:.4}'.format(f1_good_src))
    print('src_f1-multi: {:.4}'.format(f1_multi_src))
    print('---')


    # MT (only word or word&gap)
    ref_mt_tags = read_tag(join(args.ref_dir, mt_tag_fn))
    pred_mt_tags = read_tag(join(args.pred_dir, mt_tag_fn))
    f1_bad_tg, f1_good_tg, mcc_tg = compute_scores(ref_mt_tags, pred_mt_tags)
    f1_multi_tg = f1_bad_tg * f1_good_tg

    print('mt_mcc: {:.4}'.format(mcc_tg))
    print("mt_f1-bad: {:.4}".format(f1_bad_tg))
    print('mt_f1-good: {:.4}'.format(f1_good_tg))
    print('mt_f1-multi: {:.4}'.format(f1_multi_tg))
    print('---')


    # GAP
    if gap_tag_fn is not None:
        ref_gap_tags = read_tag(join(args.ref_dir, gap_tag_fn))
        pred_gap_tags = read_tag(join(args.pred_dir, gap_tag_fn))
        f1_bad_gap, f1_good_gap, mcc_gap = compute_scores(ref_gap_tags, pred_gap_tags)
        f1_multi_gap = f1_bad_gap * f1_good_gap

        print("gaps_mcc: {:.4}".format(mcc_gap))
        print("gaps_f1-bad: {:.4}".format(f1_bad_gap))
        print('gaps_f1-good: {:.4}'.format(f1_good_gap))
        print('gaps_f1-multi: {:.4}'.format(f1_multi_gap))
        print('---')



if __name__ == '__main__':
    main()