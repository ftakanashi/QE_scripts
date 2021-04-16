#!/usr/bin/env python

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to the input file.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')
    parser.add_argument('-t', '--threshold', type=float, default=0.5,
                        help='A threshold above which tag would be predicted as BAD.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.input, 'r') as f:
        lines = [l.strip() for l in f]

    new_lines = []
    for l in lines:
        probs = [float(p) for p in l.split()]
        tags = ['BAD' if p >= args.threshold else 'OK' for p in probs]
        new_lines.append(tags)

    wf = open(args.output, 'w')
    for tags in new_lines:
        wf.write(' '.join(tags) + '\n')
    wf.close()

if __name__ == '__main__':
    main()