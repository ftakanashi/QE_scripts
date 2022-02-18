#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Since the workflow for word alignment extraction by GIZA++ is complicated, this script consolidates all the steps of that workflow.
Specifically, the following commands are executedin order:

```
cp SRC_PATH src.txt; cp TGT_PATH tgt.txt
plain2snt src.txt tgt.txt
snt2cooc src.vcb tgt.vcb src_tgt.snt > src_tgt.cooc
snt2cooc tgt.vcb src.vcb tgt_src.snt > tgt_src.cooc
mkcls -psrc.txt -Vsrc.vcb.classes opt
mkcls -ptgt.txt -Vtgt.vcb.classes opt
GIZA++ -S src.vcb -T tgt.vcb -C src_tgt.snt -CoocurrenceFile src_tgt.cooc -o src2tgt -OutputPath src2tgt_dir
GIZA++ -S tgt.vcb -T src.vcb -C tgt_src.snt -CoocurrenceFile tgt_src.cooc -o tgt2src -OutputPath tgt2src_dir
```

Then, the A3.final files will be transferred into standard format of alignment files. (function a3final_to_align_lines)

```
atools from fast_align will do the symmetrization work.
atools -i forward.align -j reverse.align -c grow-diag-final-and > OUTPUT
```
'''

import argparse
import collections
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--src', type=Path,
                        help='Path to the source corpus file.')
    parser.add_argument('-t', '--tgt', type=Path,
                        help='Path to the target corpus file.')
    parser.add_argument('-o', '--output', type=Path,
                        help='Path to the output alignment file.')

    parser.add_argument('--giza_pp_dir', default='GIZA++-v2',
                        help='Path to the GIZA++ installed directory. In this directory, following binary files are '
                             'required: plain2snt.out, snt2cooc.out, GIZA++. Default: GIZA++-v2')
    parser.add_argument('--mkcls_dir', default='mkcls-v2',
                        help='Path to the mkcls installed directory. Default: mkcls-v2')
    parser.add_argument('--fast_align_dir', default='fast_align',
                        help='Path to the fast_align installed directory. Following binary files are required:'
                             'build/atools. Default: fast_align')
    # parser.add_argument('--tmp_working_dir', default='tmp_make_align',
    #                     help='A temporary working directory where all the intermediate files are saved.')

    args = parser.parse_args()
    return args


def run_cmd(cmd):
    print(cmd)
    flag = os.system(cmd)
    if flag != 0:
        raise RuntimeError(f'Error occurred runnning\n[{cmd}]')


def a3final_to_align_lines(a3final_fn, write_tmp_fn, reverse=False):
    with Path(a3final_fn).open() as f:
        a3final_lines = [l.strip() for l in f if not l.startswith('#')]

    def analyze(line):
        tokens = line.split()
        tok_i, word_i = 0, -1  # -1 stands for NULL
        align_dict = collections.defaultdict(list)

        while tok_i < len(tokens):
            if tokens[tok_i] == '({':
                tok_i += 1
                while tokens[tok_i] != '})':
                    align_dict[word_i].append(int(tokens[tok_i]) - 1)
                    tok_i += 1
                word_i += 1
            tok_i += 1

        return align_dict

    align_lines = []
    for l in a3final_lines:
        if l.startswith('NULL ({'):
            align_dict = analyze(l)
            aligns = []
            for from_i in sorted(align_dict):
                if from_i < 0: continue
                to_is = align_dict[from_i]
                if reverse:
                    aligns.extend([f'{to_i}-{from_i}' for to_i in to_is])
                else:
                    aligns.extend([f'{from_i}-{to_i}' for to_i in to_is])

            align_lines.append(' '.join(aligns))

    with Path(write_tmp_fn).open('w') as f:
        for l in align_lines:
            f.write(l + '\n')


def main():
    user_env = os.getenv('USER')
    if user_env is None:
        raise RuntimeError('You need to set the environment variable USER in order to execute GIZA++.')

    args = parse_args()

    plain2snt = os.path.join(args.giza_pp_dir, 'plain2snt.out')
    snt2cooc = os.path.join(args.giza_pp_dir, 'snt2cooc.out')
    mkcls = os.path.join(args.mkcls_dir, 'mkcls')
    giza_bin = os.path.join(args.giza_pp_dir, 'GIZA++')
    atools = os.path.join(args.fast_align_dir, 'build', 'atools')

    run_cmd(f'cp {args.src} src.txt;cp {args.tgt} tgt.txt')

    run_cmd(f'{plain2snt} src.txt tgt.txt')

    run_cmd(f'{snt2cooc} src.vcb tgt.vcb src_tgt.snt > src_tgt.cooc')
    run_cmd(f'{snt2cooc} tgt.vcb src.vcb tgt_src.snt > tgt_src.cooc')

    run_cmd(f'{mkcls} -psrc.txt -Vsrc.vcb.classes opt')
    run_cmd(f'{mkcls} -ptgt.txt -Vtgt.vcb.classes opt')

    os.makedirs('src2tgt_dir', exist_ok=True)
    os.makedirs('tgt2src_dir', exist_ok=True)

    run_cmd(f'{giza_bin} -S src.vcb -T tgt.vcb -C src_tgt.snt -CoocurrenceFile src_tgt.cooc -o src2tgt -OutputPath '
            f'src2tgt_dir')
    run_cmd(f'{giza_bin} -S tgt.vcb -T src.vcb -C tgt_src.snt -CoocurrenceFile tgt_src.cooc -o tgt2src -OutputPath '
            f'tgt2src_dir')

    a3final_to_align_lines('src2tgt_dir/src2tgt.A3.final', 'forward.align')
    a3final_to_align_lines('tgt2src_dir/tgt2src.A3.final', 'reverse.align', reverse=True)

    run_cmd(f'{atools} -i forward.align -j reverse.align -c grow-diag-final-and > {args.output}')


if __name__ == '__main__':
    main()
