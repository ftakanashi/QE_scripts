#!/usr/bin/env python

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--prefix',
                        help='Prefix for parallel corpus.')
    parser.add_argument('-s', '--src',
                        help='Suffix of source.')
    parser.add_argument('-t', '--tgt',
                        help='Suffix of target.')

    parser.add_argument('-o', '--output_dir', default='output',
                        help='Path to the output directory. Default: output')
    parser.add_argument('--pre_trained_model', default='m-bert',
                        help='Path to the pre-trained model. Default: m-bert')
    parser.add_argument('--awesome_align_dir', default='awesome-align',
                        help='Path to the awesome-align project root. Default: awesome-align')

    args = parser.parse_args()
    return args

def run_cmd(cmd):
    flag = os.system(cmd)
    if flag != 0:
        raise Exception(f'Error encountered running [{cmd}]')

def main():
    args = parse_args()

    p = args.prefix
    basefn = os.path.basename(p) + f'.{args.src}-{args.tgt}'

    # make concat file
    cmd = f'paste {p}.{args.src} {p}.{args.tgt} | awk -F \'\\t\' \'{{print $1 " ||| " $2}}\' > {basefn}'
    run_cmd(cmd)

    # run awesome-align/run_train.py
    cmd = f'python {args.awesome_align_dir}/run_train.py --output_dir {args.output_dir} --model_name_or_path ' \
          f'{args.pre_trained_model} --extraction \'softmax\' --do_train --train_mlm --train_tlm --train_tlm_full ' \
          f'--train_so --train_psi --train_data_file {basefn} --per_gpu_train_batch_size 2 ' \
          f'--gradient_accumulation_steps 4 --num_train_epochs 1 --learning_rate 2e-5 --save_steps 10000 --max_steps ' \
          f'40000 --overwrite_output_dir --overwrite_cache'
    run_cmd(cmd)

if __name__ == '__main__':
    main()