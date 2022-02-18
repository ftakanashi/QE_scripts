#!/usr/bin/env python

'''
Reverse the alignments.
Transfer all alignments from a-b to b-a.
'''

import collections
import sys

def main():
    input_lines = sys.stdin.read().strip().split('\n')
    align_pairs_lines = [
        [map(int, align.split('-')) for align in line.strip().split()]
        for line in input_lines
    ]
    for align_pairs in align_pairs_lines:
        rev_align_pairs = [(b, a) for a, b in align_pairs]
        rev_align_pairs.sort()
        print(" ".join([f"{a}-{b}" for a, b in rev_align_pairs]))

if __name__ == '__main__':
    main()