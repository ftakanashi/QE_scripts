#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-j', '--json_file', type=Path,
                        help='Path to the JSON file.')

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    with args.json_file.open() as f:
        content = json.load(f)

    pos_c = imp_c = 0

    for doc in content['data']:
        for para in doc['paragraphs']:
            for qa in para['qas']:
                if qa['is_impossible']:
                    imp_c += 1
                else:
                    pos_c += 1

    print(f'Impossible Count: {imp_c}\nPossible Count:{pos_c}\n')

if __name__ == '__main__':
    main()