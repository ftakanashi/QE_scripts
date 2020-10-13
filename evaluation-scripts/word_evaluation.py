#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
import os

from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score


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
    rev_flat_true = [1 if t == 0 else 0 for t in flat_true]
    rev_flat_pred = [1 if t == 0 else 0 for t in flat_pred]

    ok_preceision = precision_score(flat_true, flat_pred)
    ok_recall = recall_score(flat_true, flat_pred)
    bad_precision = precision_score(rev_flat_true, rev_flat_pred)
    bad_recall = recall_score(rev_flat_true, rev_flat_pred)

    bad_f1, ok_f1 = f1_score(flat_true, flat_pred, average=None, pos_label=None)

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

    return ok_preceision, ok_recall, ok_f1, bad_precision, bad_recall, bad_f1, mcc


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ref-source-tags', type=Path)
    parser.add_argument('--ref-tags', type=Path)
    parser.add_argument('--ref-gap-tags', type=Path, default=None)

    parser.add_argument('--pred-source-tags', type=Path)
    parser.add_argument('--pred-tags', type=Path)
    parser.add_argument('--pred-gap-tags', type=Path, default=None)

    # parser.add_argument('--evaluate-merged-mt', action='store_true',
    #                     help='If this option is added, then MT word tags and gap tags will be evaluated together.')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    TAG_MAP = {
        'OK': 1,
        'BAD': 0
    }

    def read_tag(fn):
        with Path(fn).open() as f:
            return [[TAG_MAP[t] for t in l.strip().split()] for l in f]

    # Source
    ref_source_tags = read_tag(args.ref_source_tags)
    pred_source_tags = read_tag(args.pred_source_tags)
    src_ok_pre, src_ok_rec, src_ok_f1, src_bad_pre, src_bad_rec, src_bad_f1, src_mcc = compute_scores(ref_source_tags,
                                                                                                      pred_source_tags)
    # f1_bad_src, f1_good_src, mcc_src = compute_scores(ref_source_tags, pred_source_tags)
    src_f1_multi = src_ok_f1 * src_bad_f1

    print(f'src_ok_precision: {src_ok_pre:.4}')
    print(f'src_ok_recall: {src_ok_rec:.4}')
    print(f'src_ok_f1: {src_ok_f1:.4}')
    print(f'src_bad_precision: {src_bad_pre:.4}')
    print(f'src_bad_recall: {src_bad_rec:.4}')
    print(f'src_bad_f1: {src_bad_f1:.4}')
    print(f'src_mcc:{src_mcc:.4}')
    print('---')

    # MT (only word or word&gap)
    ref_mt_tags = read_tag(args.ref_tags)
    pred_mt_tags = read_tag(args.pred_tags)
    mt_ok_pre, mt_ok_rec, mt_ok_f1, mt_bad_pre, mt_bad_rec, mt_bad_f1, mt_mcc = compute_scores(ref_mt_tags,
                                                                                               pred_mt_tags)
    mt_f1_multi = mt_ok_f1 * mt_bad_f1

    print(f'mt_ok_precision: {mt_ok_pre:.4}')
    print(f'mt_ok_recall: {mt_ok_rec:.4}')
    print(f'mt_ok_f1: {mt_ok_f1:.4}')
    print(f'mt_bad_precision: {mt_bad_pre:.4}')
    print(f'mt_bad_recall: {mt_bad_rec:.4}')
    print(f'mt_bad_f1: {mt_bad_f1:.4}')
    print(f'mt_mcc:{mt_mcc:.4}')
    print('---')

    # GAP
    if args.ref_gap_tags is not None and args.pred_gap_tags is not None:
        ref_gap_tags = read_tag(args.ref_gap_tags)
        pred_gap_tags = read_tag(args.pred_gap_tags)
        gap_ok_pre, gap_ok_rec, gap_ok_f1, gap_bad_pre, gap_bad_rec, gap_bad_f1, gap_mcc = compute_scores(ref_gap_tags,
                                                                                                          pred_gap_tags)
        gap_f1_multi = gap_ok_f1 * gap_bad_f1

        print(f'gap_ok_precision: {gap_ok_pre:.4}')
        print(f'gap_ok_recall: {gap_ok_rec:.4}')
        print(f'gap_ok_f1: {gap_ok_f1:.4}')
        print(f'gap_bad_precision: {gap_bad_pre:.4}')
        print(f'gap_bad_recall: {gap_bad_rec:.4}')
        print(f'gap_bad_f1: {gap_bad_f1:.4}')
        print(f'gap_mcc:{gap_mcc:.4}')
        print('---')


if __name__ == '__main__':
    main()
