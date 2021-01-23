#!/usr/bin/env python

import argparse
import os

from sklearn.metrics import precision_score, recall_score, f1_score

ORIGINAL_TAGS = ['OK', 'BAD']
FG_TAGS = ['OK', 'REP', 'INS', 'DEL']
TAG_OPTIONS = {
    'original': ORIGINAL_TAGS,
    'fine_grained': FG_TAGS
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--reference_prefix', required=True,
                        help='Path prefix of reference files.')
    parser.add_argument('-p', '--prediction_prefix', default='pred',
                        help='Path prefix of prediciton files. Default: pred')
    parser.add_argument('-m', '--mode', default='original', choices=['fine_grained', 'original'],
                        help='Select an evaluation mode.\nAvailable: fine_grained, original.\nDefault: original')

    return parser.parse_args()

def check_file_exists(args):
    assert os.path.isfile(args.reference_prefix + '.source_tags')
    assert os.path.isfile(args.prediction_prefix + '.source_tags')
    res = ['.source_tags']
    if os.path.isfile(args.prediction_prefix + '.mtword_tags'):
        assert os.path.isfile(args.reference_prefix + '.mtword_tags')
        res.append('.mtword_tags')
    if os.path.isfile(args.prediction_prefix + '.tags'):
        assert os.path.isfile(args.prediction_prefix + '.tags')
        res.append('.tags')
    return res


def compute_scores(refe_fn, pred_fn, tag_opts):
    res = {}
    ref_tags, pred_tags = [], []
    with open(refe_fn) as f:
        for line in f:
            ref_tags.extend(line.strip().split())
    with open(pred_fn) as f:
        for line in f:
            pred_tags.extend(line.strip().split())

    for tag in tag_opts:
        num_ref_tags = [1 if t == tag else 0 for t in ref_tags]
        num_pred_tags = [1 if t == tag else 0 for t in pred_tags]
        precision = precision_score(num_ref_tags, num_pred_tags)
        recall = recall_score(num_ref_tags, num_pred_tags)
        # f1 = f1_score(num_ref_tags, num_pred_tags, average='weighted', pos_label=None)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        res[tag] = {'precision': precision, 'recall': recall, 'f1': f1}
    # res['total_f1'] = f1_score(ref_tags, pred_tags, average='weighted', pos_label=None)

    return res

def main():
    args = parse_args()
    flags = check_file_exists(args)

    for type in flags:
        refe_fn = f'{args.reference_prefix}{type}'
        pred_fn = f'{args.prediction_prefix}{type}'
        tag_opts = TAG_OPTIONS[args.mode]
        res = compute_scores(refe_fn, pred_fn, tag_opts)

        print(f'======== {type}(P/R/F1) =========')
        for tag in res:
            info = res[tag]
            print('{}: {:.4} / {:.4} / {:.4}'.format(
                tag, info['precision'], info['recall'], float(info['f1'])
            ))
        # print('TOTAL F1: {}'.format(res['total_f1']))
        print('')

if __name__ == '__main__':
    main()