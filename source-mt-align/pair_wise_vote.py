#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import collections

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--files', nargs='+',
                        help='List of pair-wise result files.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    def read_fn(fn):
        with open(fn, 'r') as f:
            return [l.strip() for l in f]

    all_lines = []
    for f in args.files:
        all_lines.append(read_fn(f))

    zipped = zip(*all_lines)
    new_tags_lines = []
    for lines in zipped:
        tags_lines = [l.split() for l in lines]
        std_len = len(tags_lines[0])
        new_tags = []
        for tags_line in tags_lines:
            assert len(tags_line) == std_len
        for t_i in range(std_len):
            counter = collections.Counter([tags_line[t_i] for tags_line in tags_lines])
            new_tags.append(counter.most_common()[0][0])
        new_tags_lines.append(' '.join(new_tags))


    wf = open(args.output, 'w')
    for l in new_tags_lines:
        wf.write(l + '\n')
    wf.close()

if __name__ == '__main__':
    main()