#!/usr/bin/env python

"""
Build data file for run_mbart_prompt.py
The script firstly reads in refined MT tags and identifies REP-tagged MT words and INS-tagged MT gaps.
For MT words, we refer PE-MT alignments, for MT gaps, we refer to SRC-MT & SRC-GAP alignments
to determine where the blank should be placed in MT.
Continuous blanks are basically merged to make the data looks more natural.
BUT there are still some cases that continuous blanks cannot be merged due to some bugs.

Output file is a multi-line JSON. Each line is like:
{"translation": {"en": "A source sentence", "zh_blank": "A blanked MT [BLANK]",
"zh": "the answer sequence [ANSWER]"}}
"""

import argparse
import collections
import json

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input_prefix", type=str,
                        help="Prefix of all input files.")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to the output file.")
    parser.add_argument("--blank_token", type=str, default="¶",
                        help="Representation of [blank]. Default: '¶'")
    parser.add_argument("--answer_token", type=str, default="※",
                        help="Representation of [answer]. Default: '※'")

    parser.add_argument("--source_lang", type=str, default="en_XX",
                        help="Source language code for mBART. Default: en_XX")
    parser.add_argument("--target_lang", type=str, default="zh_CN",
                        help="Target language code for mBART. Default: zh_CN")
    parser.add_argument('--ignore_zero_blank', action='store_true', default=False,
                        help='Set this flag to ignore the samples without any blanks. Namely all MT tags are OK.')
    
    args = parser.parse_args()
    return args

def is_continous_span(spans):
    span_union = set()
    for span in spans:
        span_union = span_union.union(span)
    _max, _min = max(span_union), min(span_union)
    return _max - _min + 1 == len(span_union)

def process_step(src, mt, pe, tags, src_pe_align, mt_pe_align, gap_src_align, args):
    if type(tags) is str:
        tags = tags.strip().split()

    mt_total_len = len(tags)
    token2span = [set() for _ in range(mt_total_len)]
    for i, tag in enumerate(tags):
        if tag == "OK": continue

        if i & 1 == 0:    # gap
            target_pe_tokens = set()
            for src_tokens in gap_src_align[i]:
                target_pe_tokens = target_pe_tokens.union(src_pe_align[src_tokens])

            if len(target_pe_tokens) > 0:
                token2span[i] = target_pe_tokens

        else:    # word
            if len(mt_pe_align[(i-1)//2]) > 0:
                token2span[i] = mt_pe_align[(i-1)//2]

    if args.ignore_zero_blank and not any(token2span): return

    tgt_blank_tokens = []
    mt_tokens = mt.split()
    for i in range(mt_total_len):
        if i & 1 == 0 or tags[i] == 'DEL':
            tgt_blank_tokens.append("")
        else:
            tgt_blank_tokens.append(mt_tokens[(i-1)//2])

    i = 0
    while i < mt_total_len:
        if len(token2span[i]) == 0:
            i += 1
            continue

        j = i + 1
        while (j & 1 == 0 and len(token2span[j]) == 0) or \
              (j < mt_total_len and len(token2span[j]) > 0 and is_continous_span(token2span[i:j+1])):
            j += 1

        for pos in range(i, j-1):
            tgt_blank_tokens[pos] = ""
            token2span[pos + 1] = token2span[pos + 1].union(token2span[pos])
            token2span[pos].clear()

        tgt_blank_tokens[j-1] = args.blank_token
        i = j

    filtered_tgt_blank_tokens = [tok for tok in tgt_blank_tokens if tok != ""]
    tgt_blank_str = " ".join(filtered_tgt_blank_tokens)
    
    answer_spans = []
    pe_tokens = pe.split()
    for pe_token_span in token2span:
        if len(pe_token_span) == 0: continue
        answer_spans.append(
            " ".join([pe_tokens[i] for i in sorted(pe_token_span)])
        )
    answer_str = f" {args.answer_token} ".join(answer_spans) + f" {args.answer_token}"
    
    source_lang_short = args.source_lang.split('_')[0]
    target_lang_short = args.target_lang.split('_')[0]
    return {
        source_lang_short: src,
        f"{target_lang_short}_blank": tgt_blank_str,
        target_lang_short: answer_str
    }

def main():
    args = parse_args()

    def parse_align_lines(align_lines, reverse=False):
        res = []
        for align_line in align_lines:
            align_pairs = [map(int, align.split("-")) for align in align_line.split()]
            align_dict = collections.defaultdict(set)
            for a, b in align_pairs:
                if reverse:
                    align_dict[b].add(a)
                else:
                    align_dict[a].add(b)
            res.append(align_dict)
        return res
    
    def read_fn(suf):
        fn = f"{args.input_prefix}.{suf}"
        with open(fn) as f:
            return [l.strip() for l in f]

    src_lines = read_fn("src")
    mt_lines = read_fn("mt")
    pe_lines = read_fn("pe")
    mt_tags = read_fn("refine.tags")

    src_pe_aligns = parse_align_lines(read_fn("src-pe.align"))
    mt_pe_aligns = parse_align_lines(read_fn("pe-mt.align"), reverse=True)
    gap_src_aligns = parse_align_lines(read_fn("src-gap.align"), reverse=True)

    # simple sanity check
    std_len = len(src_lines)
    for lines in (mt_lines, pe_lines, mt_tags, src_pe_aligns, mt_pe_aligns, gap_src_aligns):
        assert len(lines) == std_len, f"Line number does not match."

    rows = []
    valid_cnt = 0
    for i in range(std_len):
        row_res = process_step(
            src_lines[i],
            mt_lines[i],
            pe_lines[i],
            mt_tags[i],
            src_pe_aligns[i],
            mt_pe_aligns[i],
            gap_src_aligns[i],
            args
        )

        if row_res is None: continue

        row_str = json.dumps({
            'translation': row_res
        }, ensure_ascii=False)
        rows.append(row_str)
        valid_cnt += 1

    wf = open(args.output, "w")
    for row in rows:
        wf.write(row + "\n")

    print(f"Among {std_len} samples, {valid_cnt} samples are dumped.")

if __name__ == '__main__':
    main()