#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input',
                        help='Path to the MT word tags file to insert OK gap tags.')
    parser.add_argument('-o', '--output',
                        help='Path to the output file.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.input, 'r') as f:
        lines = [l.strip() for l in f]

    wf = open(args.output, 'w')
    for l in lines:
        wf.write(' '.join(['OK'] * (len(l.split()) + 1) ) + '\n')

    wf.close()

if __name__ == '__main__':
    main()