#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
    Transform original OK/BAD word tags into fine-grained OK/REP/INS/DEL
    source_tags, mtword_tags and alignments are needed.

    --remove-mismatch-align   add this flag to remove the alignments with mismatch (OK-BAD) then generate a modified
    alignment file.
'''

import argparse
import collections
import copy
import os

from pathlib import Path

class MismatchStrategy:
    def __init__(self):
        self.nonAlignSrcOkCount = 0
        self.nonAlignMtOkCount = 0
        self.mismatchTagCount = 0

    def modify(self, src_tags, mt_tags, alignments, **kwargs):
        for t in src_tags:
            assert t in ('OK', 'BAD')
        for t in mt_tags:
            assert t in ('OK', 'BAD')

        mod_src_tags, mod_mt_tags, mod_align = self._modify(src_tags, mt_tags, alignments, **kwargs)

        assert len(mod_src_tags) == len(src_tags)
        assert len(mod_mt_tags) == len(mt_tags)
        for t in mod_src_tags:
            assert t in ('OK', 'REP', 'INS', 'DEL')
        for t in mod_mt_tags:
            assert t in ('OK', 'REP', 'INS', 'DEL')
        return mod_src_tags, mod_mt_tags, mod_align

    def _modify(self, src_tags, mt_tags, alignments, **kwargs):
        '''
        need to be implemented
        '''
        pass

class RemoveStrategy(MismatchStrategy):
    def __init__(self):
        super(RemoveStrategy, self).__init__()

    def _modify(self, src_tags, mt_tags, alignments, **kwargs):

        for si in range(len(src_tags)):
            for ti in range(len(mt_tags)):
                if (si, ti) in alignments and src_tags[si] != mt_tags[ti]:
                    self.mismatchTagCount += 2
                    alignments.remove((si, ti))

        s2t_dict = collections.defaultdict(list)
        t2s_dict = collections.defaultdict(list)
        for a, b in alignments:
            s2t_dict[a].append(b)
            t2s_dict[b].append(a)

        mod_src_tags = []
        mod_mt_tags = []
        for si, src_tag in enumerate(src_tags):
            if src_tag == 'BAD':
                if len(s2t_dict[si]) == 0:
                    mod_src_tags.append('INS')
                else:
                    mod_src_tags.append('REP')
            else:
                if len(s2t_dict[si]) == 0: self.nonAlignSrcOkCount += 1
                mod_src_tags.append('OK')

        for ti, mt_tag in enumerate(mt_tags):
            if mt_tag == 'BAD':
                if len(t2s_dict[ti]) == 0:
                    mod_mt_tags.append('DEL')
                else:
                    mod_mt_tags.append('REP')
            else:
                if len(t2s_dict[ti]) == 0: self.nonAlignMtOkCount += 1
                mod_mt_tags.append('OK')

        mod_aligns = []
        if kwargs.get('output_alignment', False):
            for si in s2t_dict:
                for ti in s2t_dict[si]:
                    mod_aligns.append(f'{si}-{ti}')

        return mod_src_tags, mod_mt_tags, mod_aligns


OK_LABELS = ['OK']
BAD_LABELS = ['BAD', 'REP', 'INS', 'DEL']

STRATEGY_MAP = {
    'remove': RemoveStrategy
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source_tags', type=Path,
                        help='Path to the source tag files.')
    parser.add_argument('-t', '--mt_tags', type=Path,
                        help='Path to the MT tag files.\nNEEDS ONLY WORD TAG FILE!!!')
    parser.add_argument('-a', '--source_mt_align', type=Path,
                        help='Path to the source-MT alignment file.')
    parser.add_argument('-o', '--output_dir', default=os.getcwd(),
                        help='Path to the output directory where modified tag files and other output files are saved.')

    parser.add_argument('--output_midfix', default='refine',
                        help='Midfix in the filename of output files.')

    parser.add_argument('--mismatch_strategy', default='remove', choices=['remove'],
                        help='Set the strategy handling unexpected situations like OK-BAD alignments or non-aligned '
                             'OKs.\nYou need to implement a MismatchStrategy class in code.\nDefault: ignore_mismatch.')
    parser.add_argument('--output_cleaned_alignment', action='store_true', default=False,
                        help='Add this flag to re-generate a cleaned alignment file which has no mismatch according '
                             'to the given tags.')

    args = parser.parse_args()

    return args


def modify_tags_and_aligns(source_tags, mt_tags, align_dict, reversed_align_dict):
    # the source word is not translated, thus INSERT is required
    for from_i, source_tag in enumerate(source_tags):
        if len(align_dict[from_i]) == 0 and source_tag == 'BAD':
            source_tags[from_i] = 'INS'

    # the MT word is redundant, thus DELETE is required
    for to_j, mt_tag in enumerate(mt_tags):
        if len(reversed_align_dict[to_j]) == 0 and mt_tag == 'BAD':
            mt_tags[to_j] = 'DEL'

    for from_i, source_tag in enumerate(source_tags):
        dummy_to_js = copy.deepcopy(align_dict[from_i])
        for to_j in dummy_to_js:
            mt_tag = mt_tags[to_j]

            if source_tag == 'OK' and mt_tag == 'OK':
                continue

            elif (source_tag in BAD_LABELS and mt_tag in BAD_LABELS):
                source_tags[from_i] = 'REP'
                mt_tags[to_j] = 'REP'

            elif (source_tag == 'OK' and mt_tag in BAD_LABELS):
                align_dict[from_i].remove(to_j)
                reversed_align_dict[to_j].remove(from_i)
                if len(reversed_align_dict[to_j]) == 0:
                    # when there is no available alignment rest for this MT word
                    mt_tags[to_j] = 'DEL'

            elif (source_tag in BAD_LABELS and mt_tag == 'OK'):
                align_dict[from_i].remove(to_j)
                reversed_align_dict[to_j].remove(from_i)
                if len(align_dict[from_i]) == 0:
                    # when there is no available alignment rest for this source word
                    source_tags[from_i] = 'INS'

            else:
                raise RuntimeError('Not recognized situation.')


def main():
    args = parse_args()

    def read_fn(fn):
        with fn.open() as f:
            return [l.strip() for l in f]

    source_tag_lines = read_fn(args.source_tags)
    mt_tag_lines = read_fn(args.mt_tags)
    align_lines = read_fn(args.source_mt_align)

    strategy = STRATEGY_MAP[args.mismatch_strategy]()

    mod_src_tag_lines = []
    mod_mt_tag_lines = []
    mod_align_lines = []
    for source_tag_line, mt_tag_line, align_line in zip(source_tag_lines, mt_tag_lines, align_lines):
        src_tags = source_tag_line.split()
        mt_tags = mt_tag_line.split()
        alignments = set()
        for aligns in align_line.split():
            a, b = map(int, aligns.split('-'))
            alignments.add((a, b))

        mod_src_tags, mod_mt_tags, mod_aligns = strategy.modify(src_tags, mt_tags, alignments,
                                                                output_alignment=args.output_cleaned_alignment)

        mod_src_tag_lines.append(' '.join(mod_src_tags))
        mod_mt_tag_lines.append(' '.join(mod_mt_tags))
        if len(mod_aligns) > 0:
            mod_align_lines.append(' '.join(mod_aligns))


    def get_new_fn(fn):
        # dn = os.path.dirname(os.path.abspath(fn))
        dn = args.output_dir
        fn = os.path.basename(fn)
        prefix, ext = fn.rsplit('.', 1)
        return os.path.join(dn, f'{prefix}.{args.output_midfix}.{ext}')

    # write in new source tags
    with open(get_new_fn(args.source_tags), 'w') as f:
        for l in mod_src_tag_lines:
            f.write(l + '\n')

    # write in new mt tags
    with open(get_new_fn(args.mt_tags), 'w') as f:
        for l in mod_mt_tag_lines:
            f.write(l + '\n')

    if args.output_cleaned_alignment:
        # write in new alignment
        with open(get_new_fn(args.source_mt_align), 'w') as f:
            for l in mod_align_lines:
                f.write(l + '\n')

if __name__ == '__main__':
    main()
