#!/usr/bin/env python

import argparse
import collections
import numpy as np
import os
import random
import warnings

from bisect import bisect_left, insort_left
from tqdm import tqdm


class SentAlignment:
    def __init__(self, src_tokens, src_tags, mt_tokens, mt_word_tags, mt_gap_tags, align_line=None, **kwargs):
        self.src_tokens = src_tokens
        self.src_tags = src_tags
        assert len(self.src_tags) == len(self.src_tokens)

        self.mt_tokens = mt_tokens
        self.mt_word_tags = mt_word_tags
        self.mt_gap_tags = mt_gap_tags
        assert len(self.mt_tokens) == len(self.mt_word_tags)
        assert len(self.mt_tokens) + 1 == len(self.mt_gap_tags)

        self.s2t = [set() for _ in range(len(self.src_tokens))]
        self.s2gap = [set() for _ in range(len(self.src_tokens))]
        self.t2s = [set() for _ in range(len(self.mt_tokens))]
        self.gap2s = [set() for _ in range((1 + len(self.mt_tokens)))]

        if align_line:
            self.load_word_align_line(align_line)

        self.target_vocab = kwargs.get('target_vocab', None)

    def load_word_align_line(self, align_line):

        for align_token in align_line.strip().split():
            a, b = list(map(int, align_token.split('-')))
            self.s2t[a].add(b)
            self.t2s[b].add(a)

    def drop_source_word(self, s_i):
        assert 0 <= s_i < len(self.src_tokens), f'Invalid index of source word: {s_i}.'
        self.src_tokens.pop(s_i)
        self.src_tags.pop(s_i)

        aligned_t_is = self.s2t.pop(s_i)
        aligned_g_is = self.s2gap.pop(s_i)

        for t_i in range(len(self.mt_tokens)):
            if len(self.t2s[t_i]) == 0: continue
            new_s_is = set()
            for old_s_i in self.t2s[t_i]:
                if old_s_i < s_i:
                    new_s_is.add(old_s_i)
                elif old_s_i == s_i:
                    continue
                else:
                    new_s_is.add(old_s_i - 1)
            self.t2s[t_i] = new_s_is

        for t_i in aligned_t_is:
            if len(self.t2s[t_i]) == 0:
                self.mt_word_tags[t_i] = 'DEL'

        for g_i in range(1 + len(self.mt_tokens)):
            if len(self.gap2s[g_i]) == 0: continue
            new_s_is = set()
            for old_s_i in self.gap2s[g_i]:
                if old_s_i < s_i:
                    new_s_is.add(old_s_i)
                elif old_s_i == s_i:
                    continue
                else:
                    new_s_is.add(old_s_i - 1)
            self.gap2s[g_i] = new_s_is

        for g_i in aligned_g_is:
            if len(self.gap2s[g_i]) == 0:
                self.mt_gap_tags[g_i] = 'OK'

    def drop_mt_word(self, t_i):
        assert 0 <= t_i < len(self.mt_tokens), f'Invalid index of target word: {t_i}.'
        self.mt_tokens.pop(t_i)
        self.mt_word_tags.pop(t_i)

        mt_aligned_source = self.t2s.pop(t_i)
        # gap_aligned_source = self.gap2s.pop(t_i)

        for s_i in range(len(self.src_tokens)):
            if len(self.s2t[s_i]) == 0: continue
            new_t_is = set()
            for old_t_i in self.s2t[s_i]:
                if old_t_i < t_i:
                    new_t_is.add(old_t_i)
                elif old_t_i == t_i:
                    continue
                else:
                    new_t_is.add(old_t_i - 1)
            self.s2t[s_i] = new_t_is

        for s_i in mt_aligned_source:
            if len(self.s2t[s_i]) == 0:
                self.src_tags[s_i] = 'INS'
                self.mt_gap_tags[t_i] = 'INS'
                self.s2gap[s_i].add(t_i)
                self.gap2s[t_i].add(s_i)

        for s_i in range(len(self.src_tokens)):
            if len(self.s2gap[s_i]) == 0: continue
            new_g_is = set()
            for old_g_i in self.s2gap[s_i]:
                if old_g_i <= t_i:
                    new_g_is.add(old_g_i)
                else:
                    new_g_is.add(old_g_i - 1)
            self.s2gap[s_i] = new_g_is

        gap_aligned_source = self.gap2s.pop(t_i)
        self.gap2s[t_i] = self.gap2s[t_i].union(gap_aligned_source)
        old_tag = self.mt_gap_tags.pop(t_i)
        if not (old_tag == 'OK' and self.mt_gap_tags[t_i] == 'OK'):
            self.mt_gap_tags[t_i] = 'INS'

    def replace_word_pair(self, s_i, t_i):
        assert 0 <= s_i < len(self.src_tokens), f'Invalid index of source word: {s_i}'
        assert 0 <= t_i < len(self.mt_tokens), f'Invalid index of target word: {t_i}.'
        assert t_i in self.s2t[s_i] and s_i in self.t2s[
            t_i], f'source word [{s_i}] and MT word [{t_i}] are not aligned.'
        self.src_tags[s_i] = 'REP'
        self.mt_word_tags[t_i] = 'REP'
        self.mt_tokens[t_i] = random.choice(self.target_vocab)

    def get_source_line(self):
        return ' '.join(self.src_tokens)

    def get_mt_line(self):
        return ' '.join(self.mt_tokens)

    def get_source_tags(self):
        return ' '.join(self.src_tags)

    def get_mt_word_tags(self):
        return ' '.join(self.mt_word_tags)

    def get_mt_gap_tags(self):
        return ' '.join(self.mt_gap_tags)

    def get_align_line(self, category='word'):
        assert category in ('word', 'gap'), f'Invalid category {category}.'
        res = []
        if category == 'word':
            for s_i in range(len(self.s2t)):
                for t_i in self.s2t[s_i]:
                    assert s_i in self.t2s[t_i]
                    res.append(f'{s_i}-{t_i}')
        elif category == 'gap':
            for s_i in range(len(self.s2gap)):
                for g_i in self.s2gap[s_i]:
                    assert s_i in self.gap2s[g_i]
                    res.append(f'{s_i}-{g_i * 2}')

        return ' '.join(res)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', help='Path to the source corpus.')
    parser.add_argument('-t', '--tgt', help='Path to the target(PE) corpus.')
    parser.add_argument('-a', '--align', help='Path to the alignment file. (EXPECTED TO BE ORDERED)')
    parser.add_argument('-op', '--output_prefix', help='Prefix (including directory) of all output files.')

    parser.add_argument('--replace_prob', type=float, default=0.3,
                        help='Probability of replacement noise.')
    parser.add_argument('--drop_source_prob', type=float, default=0.1,
                        help='Probability of drop source word noise.')
    parser.add_argument('--drop_target_prob', type=float, default=0.1,
                        help='Probability of drop target word noise.')

    parser.add_argument('--target_vocab', default=None, help='Provide the vocabulary file of target language for '
                                                             'replacement noise. If None, a vocabulary is automatically '
                                                             'generated based on the provided target corpus.')

    parser.add_argument('--default_source_tag', type=str,
                        default='OK', help='Default tag for non-aligned source words.')
    parser.add_argument('--default_mt_word_tag', type=str,
                        default='OK', help='Default tag for non-aligned PE words.')

    args = parser.parse_args()

    assert args.replace_prob + args.drop_source_prob + args.drop_target_prob <= 1.0

    return args


def parse_alignment(align_line, rev=False):
    align_dict = collections.defaultdict(list)
    for align_pair_s in align_line.strip().split():
        a, b = list(map(int, align_pair_s.split('-')))
        if not rev:
            align_dict[a].append(b)
        else:
            align_dict[b].append(a)
    return align_dict


def main():
    args = parse_args()

    def read_fn(fn):
        print(f'Reading {fn}...')
        with open(fn) as f:
            return [l.strip() for l in f]

    src_lines = read_fn(args.src)
    tgt_lines = read_fn(args.tgt)
    align_lines = read_fn(args.align)

    print('\n' + '=' * 20 + '\n')

    if args.target_vocab is not None:
        target_vocab = read_fn(args.target_vocab)
    else:
        print('Generating target vocab...')
        target_vocab = set()
        for l in tgt_lines:
            for tgt_token in l.split():
                target_vocab.add(tgt_token)
        target_vocab = list(target_vocab)

    std_len = len(src_lines)
    for lines in (tgt_lines, align_lines):
        assert len(lines) == std_len, f'Unmatched number of lines'

    retain_cnt = replace_noise_cnt = drop_source_noise_cnt = drop_target_noise_cnt = 0
    new_src_lines, new_tgt_lines = [], []
    new_source_tags, new_mt_word_tags, new_mt_gap_tags = [], [], []
    new_src_mt_align, new_src_gap_align = [], []

    for i, (src_line, tgt_line, align_line) in enumerate(zip(src_lines, tgt_lines, align_lines)):
        try:
            src_tokens = src_line.split()
            tgt_tokens = tgt_line.split()
            src_tags = ['OK' for _ in range(len(src_tokens))]
            mt_word_tags = ['OK' for _ in range(len(tgt_tokens))]
            mt_gap_tags = ['OK' for _ in range(len(tgt_tokens) + 1)]

            align_dict = parse_alignment(align_line)
            rev_align_dict = parse_alignment(align_line, rev=True)
            if args.default_source_tag != 'OK':
                for s_i in range(len(src_tokens)):
                    if len(align_dict[s_i]) == 0: src_tags[s_i] = args.default_source_tag
            if args.default_mt_word_tag != 'OK':
                for t_i in range(len(tgt_tokens)):
                    if len(rev_align_dict[t_i]) == 0: mt_word_tags[t_i] = args.default_mt_word_tag

            sent_pair = SentAlignment(src_tokens, src_tags, tgt_tokens, mt_word_tags, mt_gap_tags, align_line,
                                      target_vocab=target_vocab)

            ope_distrib = [
                1 - args.replace_prob - args.drop_source_prob - args.drop_target_prob,
                args.replace_prob, args.drop_source_prob, args.drop_target_prob
            ]

            dropped_source = []
            dropped_target = []

            for align_pair_str in align_line.split():

                a, b = list(map(int, align_pair_str.split('-')))
                # 目前只对一对一对应导入noise 简化代码情况。要不然太复杂了
                if not (align_dict[a] == [b,] and rev_align_dict[b] == [a,]): continue
                ope = np.random.choice(4, 1, p=ope_distrib)

                dropped_prev_source_cnt = bisect_left(dropped_source, a)
                dropped_prev_target_cnt = bisect_left(dropped_target, b)

                if ope == 0:
                    retain_cnt += 1
                    continue

                elif ope == 1:
                    # print(f'REP: {a}-{b}')
                    sent_pair.replace_word_pair(
                        a - dropped_prev_source_cnt,
                        b - dropped_prev_target_cnt
                    )
                    replace_noise_cnt += 1

                elif ope == 2:
                    # print(f'DROP SOURCE: {a}')
                    sent_pair.drop_source_word(a - dropped_prev_source_cnt)
                    insort_left(dropped_source, a)
                    drop_source_noise_cnt += 1

                elif ope == 3:
                    # print(f'DROP MT: {b}')
                    sent_pair.drop_mt_word(b - dropped_prev_target_cnt)
                    insort_left(dropped_target, b)
                    drop_target_noise_cnt += 1

                else:
                    raise ValueError(f'Invalid operation.')

            new_src_lines.append(sent_pair.get_source_line())
            new_tgt_lines.append(sent_pair.get_mt_line())
            new_source_tags.append(sent_pair.get_source_tags())
            new_mt_word_tags.append(sent_pair.get_mt_word_tags())
            new_mt_gap_tags.append(sent_pair.get_mt_gap_tags())
            new_src_mt_align.append(sent_pair.get_align_line('word'))
            new_src_gap_align.append(sent_pair.get_align_line('gap'))

        except Exception as e:
            warnings.warn(f'Problem occurred processing line.{i}')
            raise e

    print(f'Among all tokens,\nRetained Tokens: {retain_cnt}\nReplaced Tokens: {replace_noise_cnt}\n'
          f'Dropped Source Tokens: {drop_source_noise_cnt}\nDropped Target Tokens: {drop_target_noise_cnt}')

    def write_fn(fn, lines):
        print(f'Writing {os.path.basename(fn)}...')
        with open(fn, 'w') as wf:
            for l in lines:
                wf.write(l.strip() + '\n')

    print('\n' + '=' * 20 + '\n')
    for ext, lines in [('src', new_src_lines), ('mt', new_tgt_lines),
                       ('source_tags', new_source_tags), ('mtword_tags', new_mt_word_tags),
                       ('gap_tags', new_mt_gap_tags),
                       ('src-mt.align', new_src_mt_align), ('src-gap.align', new_src_gap_align)]:
        write_fn(args.output_prefix + f'.{ext}', lines)


def exp_main():
    source_tokens = 'A B C E'.split()
    source_tags = 'OK OK OK OK'.split()
    mt_tokens = 'a c b d'.split()
    mt_word_tags = 'OK OK OK OK'.split()
    mt_gap_tags = 'OK OK OK OK OK'.split()
    align = '0-0 1-2 2-1'

    sent_pair = SentAlignment(source_tokens, source_tags, mt_tokens, mt_word_tags, mt_gap_tags, align,
                              keep_source_gap_align=True)

    sent_pair.drop_mt_word(1)
    sent_pair.drop_source_word(0)

    print(sent_pair.get_source_tags())
    print(sent_pair.get_source_line())
    print(sent_pair.get_align_line('word'))
    print(sent_pair.get_align_line('gap'))
    print(sent_pair.get_mt_line())
    print(sent_pair.get_mt_word_tags())
    print(sent_pair.get_mt_gap_tags())


if __name__ == '__main__':
    # exp_main()
    main()
