#!/usr/bin/env python

'''
This script sum up all results under different blank-mask ratio and output a jointly evaluated result.
It simply read content in all match_rate.txt and sum up all top-1 & top-n counts as the total matched count.

Example of usage:
python <this script> -d results.m*.n10 --output_dir results.joint.n10
'''

import argparse
# import glob
import os
import re

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", nargs="+",
                        help="A list of dirs containing sub-results you want to evaluate. Support wildcard(*).")
    parser.add_argument("--output_dir", default=os.path.join(os.getcwd(), "results.jointly"),
                        help="A directory to save the output file.")

    args = parser.parse_args()
    return args

def parse_content(fn):
    with open(fn) as f:
        content = f.read().strip()

    match = re.search("Top 1 Match.+?\((\d+) / (\d+)\)\nTop n Match.+?\((\d+) / (\d+)\)", content)
    if match is None:
        raise SyntaxError(f"Cannot identify top-1 & top-n match in the content of {fn}")
    top_1_match = int(match.group(1))
    total_cnt_a = int(match.group(2))
    top_n_match = int(match.group(3))
    total_cnt_b = int(match.group(4))
    assert total_cnt_a == total_cnt_b, f"Total blank numbers is inconsistent inside {fn}"

    return top_1_match, top_n_match, total_cnt_b

def main():
    args = parse_args()
    subdirs = args.directory

    # file existence sanity check
    for subdir in subdirs:
        assert os.path.isfile(os.path.join(subdir, "match_rate.txt")), f"Cannot find match_rate.txt in {subdir}"

    total_top_1_cnt = total_top_n_cnt = 0
    total_cnt = -1
    for subdir in subdirs:
        fn = os.path.join(subdir, "match_rate.txt")
        top_1_cnt, top_n_cnt, part_total_cnt = parse_content(fn)
        if total_cnt < 0:
            total_cnt = part_total_cnt
        else:
            assert total_cnt == part_total_cnt, f"Total blank number is inconsistent with file {fn}"
        total_top_1_cnt += top_1_cnt
        total_top_n_cnt += top_n_cnt

    msg = f"Jointly evaluated {', '.join(subdirs)}.\n"
    msg += f"Top 1 Match: {total_top_1_cnt / total_cnt:.4f} ({total_top_1_cnt} / {total_cnt})\n" \
           f"Top n Match: {total_top_n_cnt / total_cnt:.4f} ({total_top_n_cnt} / {total_cnt})\n"

    print(msg)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "match_rate.txt"), "w") as wf:
        wf.write(msg)

if __name__ == '__main__':
    main()