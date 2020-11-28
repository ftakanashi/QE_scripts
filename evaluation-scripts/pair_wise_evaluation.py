#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
from sklearn.metrics import precision_score, recall_score, f1_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prediction',
                        help='Path to the prediction file.')
    parser.add_argument('-r', '--reference',
                        help='Path to the reference file.')
    parser.add_argument('-v', '--valid_tags', nargs='+',
                        help='List of valid tags in reference that need to be evaluated.')

    return parser.parse_args()

def main():
    args = parse_args()

    def read_fn(fn):
        with open(fn, 'r') as f:
            return [l.strip().split() for l in f]

    pred_tags_lines = read_fn(args.prediction)
    ref_tags_lines = read_fn(args.reference)

    pred_tags = []
    ref_tags = []
    for pred_tags_line, ref_tags_line in zip(pred_tags_lines, ref_tags_lines):
        assert len(pred_tags_line) == len(ref_tags_line)
        for pred_tag, ref_tag in zip(pred_tags_line, ref_tags_line):
            if ref_tag in args.valid_tags:
                pred_tags.append(pred_tag)
                ref_tags.append(ref_tag)

    def evaluate_tag(tag):
        tag2id = {t: 0 for t in args.valid_tags}
        tag2id[tag] = 1

        pred_ids = [tag2id[t] for t in pred_tags]
        ref_ids = [tag2id[t] for t in ref_tags]
        precision = precision_score(ref_ids, pred_ids, zero_division=True)
        recall = recall_score(ref_ids, pred_ids, zero_division=True)
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f'Result for {tag}:\nPrecision/Recall/F1: {precision:.4}/{recall:.4}/{f1:.4}\n')

    for t in args.valid_tags:
        evaluate_tag(t)

if __name__ == '__main__':
    main()