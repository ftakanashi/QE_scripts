#!/usr/bin/env python

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prediction_file',
                        help='Path to the predictions_.json produced by SQuAD prediction model.')
    parser.add_argument('-ao', '--alignment_output',
                        help='Path to the alignment output file containing the alignment information between source '
                             'words and MT gaps.')
    parser.add_argument('-to', '--tag_output',
                        help='Path to the gap tag output file. If a gap is not aligned to any source words, '
                             'it would be regarded as OK otherwise BAD.')

    parser.add_argument('--align_prob_threshold', default=0.5, type=float,
                        help='A probability threshold for extracting alignment. Note that currently, ')


