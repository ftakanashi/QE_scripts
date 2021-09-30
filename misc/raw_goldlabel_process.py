#!/usr/bin/env python

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to the input file (raw gold labels data).')
    parser.add_argument('-o', '--output', help='Path to the output file.')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    rf = open(args.input, 'r')
    wf = open(args.output, 'w')
    curr_sent = 0
    stack = []
    for line in rf:
        info = line.strip().split()
        sent_id = int(info[3])
        tag = info[6]
        if sent_id == curr_sent:
            stack.append(tag)
        else:
            wf.write(' '.join(stack) + '\n')
            stack.clear()
            stack.append(tag)
            curr_sent = sent_id

    if len(stack) > 0:
        wf.write(' '.join(stack) + '\n')

    rf.close()
    wf.close()

if __name__ == '__main__':
    main()