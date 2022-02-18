#!/usr/bin/env python

'''
This script takes a MT word tag file as an input.
Then insert OK into all gaps between the word tags as MT gap tags.
'''

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
        word_tags = l.split()
        res = ['OK', ]
        for wt in word_tags:
            res.append(wt)
            res.append('OK')
        wf.write(' '.join(res) + '\n')

    wf.close()

if __name__ == '__main__':
    main()