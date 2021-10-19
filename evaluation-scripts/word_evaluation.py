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

    parser.add_argument('--simple', action='store_true', default=False,
                        help="Set this flag to output the simplest results. Like source MCC & MT MCC only.")

    return parser.parse_args()


def check_file_exists(args):
    # assert os.path.isfile(args.reference_prefix + '.source_tags')
    # assert os.path.isfile(args.prediction_prefix + '.source_tags')
    res = []
    if os.path.isfile(args.prediction_prefix + '.source_tags'):
        assert os.path.isfile(args.reference_prefix + '.source_tags')
        res.append('.source_tags')
    if os.path.isfile(args.prediction_prefix + '.mtword_tags'):
        assert os.path.isfile(args.reference_prefix + '.mtword_tags')
        res.append('.mtword_tags')
    if os.path.isfile(args.prediction_prefix + '.gap_tags'):
        assert os.path.isfile(args.reference_prefix + '.gap_tags')
        res.append('.gap_tags')
    if os.path.isfile(args.prediction_prefix + '.tags'):
        assert os.path.isfile(args.prediction_prefix + '.tags')
        res.append('.tags')
    return res


def compute_scores(refe_fn, pred_fn, tag_opts, args):
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
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        res[tag] = {'precision': precision, 'recall': recall, 'f1': f1}

    tag_map = {}
    tag_i = 0
    for ref_tag in ref_tags:
        if ref_tag not in tag_map:
            tag_map[ref_tag] = tag_i
            tag_i += 1

    num_ref_tags = [tag_map[t] for t in ref_tags]
    num_pred_tags = [tag_map.get(t, tag_i) for t in pred_tags]
    total_f1 = f1_score(num_ref_tags, num_pred_tags, average='weighted', pos_label=None)

    # if only two types of tags are predicted, add MCC
    if len(tag_opts) == 2:
        pos_tag, nev_tag = tag_opts
        tp = fp = tn = fn = 0
        for pred_tag, ref_tag in zip(pred_tags, ref_tags):
            if pred_tag == pos_tag:
                if ref_tag == pos_tag:
                    tp += 1
                else:
                    fp += 1
            else:
                if ref_tag == pos_tag:
                    fn += 1
                else:
                    tn += 1

        mcc_numerator = (tp * tn) - (fp * fn)
        mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        mcc = mcc_numerator / (mcc_denominator + 1e-5)
        res['mcc'] = mcc

    return res, total_f1


def main():
    args = parse_args()
    flags = check_file_exists(args)

    for type in flags:
        refe_fn = f'{args.reference_prefix}{type}'
        pred_fn = f'{args.prediction_prefix}{type}'
        tag_opts = TAG_OPTIONS[args.mode]
        res, total_f1 = compute_scores(refe_fn, pred_fn, tag_opts, args)

        if args.simple:
            assert args.mode == 'original', 'Only original mode is supported currently.'
            print(f'{type}: {res["mcc"]}')

        else:

            print(f'======== {type}(P/R/F1) =========')
            for tag in res:
                if tag != 'mcc':
                    info = res[tag]
                    print('{}: {:.4} / {:.4} / {:.4}'.format(
                        tag, info['precision'], info['recall'], info['f1']
                    ))
            print('TOTAL F1: {:.4}'.format(total_f1))
            if 'mcc' in res:
                print('MCC: {:.4}'.format(res['mcc']))
            print('')


if __name__ == '__main__':
    main()