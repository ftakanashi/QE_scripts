#!/usr/bin/env python

NOTE = \
    '''
        This script helps you to search the best threshold discriminating OK/BAD tags.
        You need a *.prob file in which every item is a probability of corresponding word to be BAD.
        You also need a label file.
        
        Usage is like:
        python search_best_tag_threshold.py -r xxx/test -p xxx/pred --min_ths 0.0 --max_ths 1.0 --stride 0.01
        --verbose
    '''

import argparse
import datetime
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--ref_prefix',
                        help='Prefix path of the reference files.')
    parser.add_argument('-p', '--pred_prefix',
                        help='Prefix path of the prediction files.')

    parser.add_argument('--min_ths', type=float, default=0.0,
                        help='Lower bound of range to search.')
    parser.add_argument('--max_ths', type=float, default=1.0,
                        help='Higher bound of range to search')
    parser.add_argument('--stride', type=float, default=0.01,
                        help='Stride for searching. DO NOT set it too small.')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Set this flag to output more detailed process of search.')
    parser.add_argument('--no_output', action='store_true', default=False,
                        help='Set this flag to do threshold search only. No output files will be generated')
    parser.add_argument('--use_predicted_gap', action='store_true', default=False,
                        help='Set this flag to use really predicted gap tags rather than "All OK" when merging '
                             'gap/mtword tags.')

    parser.add_argument('--script_root', type=str, default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help='Root path for QE_scripts.')

    args = parser.parse_args()
    args.min_ths = max(0.0, min(1.0, args.min_ths))
    args.max_ths = max(0.0, min(1.0, args.max_ths))
    assert args.min_ths <= args.max_ths, f'Invalid threshold range ({args.min_ths}, {args.max_ths}).'
    assert args.stride >= 0.001, f'Stride {args.stride} too small.'
    return args


def calc_scores(ref_tags, pred_tags):
    pos_tag, nev_tag = 'OK', 'BAD'
    tp = fp = tn = fn = 0
    for pred_tag, ref_tag in zip(pred_tags, ref_tags):
        if pred_tag == pos_tag:
            if ref_tag == pos_tag:
                tp += 1
            else:
                fp += 1
        else:
            if ref_tag == pos_tag:
                fn += 1
            else:
                tn += 1

    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = mcc_numerator / (mcc_denominator + 1e-5)

    ok_precision = tp / (tp + fp)
    ok_recall = tp / (tp + fn)
    ok_f1 = 2 * ok_precision * ok_recall / (ok_precision + ok_recall)

    bad_precision = tn / (tn + fn)
    bad_recall = tn / (tn + fp)
    bad_f1 = 2 * bad_precision * bad_recall / (bad_precision + bad_recall)

    return {
        'mcc': mcc,
        'ok_f1': ok_f1,
        'ok_p': ok_precision,
        'ok_r': ok_recall,
        'bad_f1': bad_f1,
        'bad_p': bad_precision,
        'bad_r': bad_recall
    }


def process(pred_fn, ref_fn, args):
    if args.verbose:
        print(f'Processing {pred_fn}...')

    all_ref_tags = []
    with open(ref_fn) as f:
        for line in f:
            line_tags = line.strip().split()
            all_ref_tags.extend(line_tags)

    all_pred_probs = []
    with open(pred_fn) as f:
        for line in f:
            line_probs = [float(f) for f in line.strip().split()]
            all_pred_probs.extend(line_probs)

    ths = args.min_ths
    max_mcc, opt_ths = -99999, -1
    total_cnt = max(1, int((args.max_ths - args.min_ths) / args.stride))
    cnt = 0
    while ths <= args.max_ths:
        if args.verbose:
            print('[{:.0f}%]'.format((cnt / total_cnt) * 100), end='')
            print(f'threshold = {ths:.4f},', end='\t')

        all_dummy_pred_tags = ['OK' if p < ths else 'BAD' for p in all_pred_probs]
        res = calc_scores(all_ref_tags, all_dummy_pred_tags)
        mcc = res['mcc']
        if mcc > max_mcc:
            max_mcc = mcc
            opt_ths = ths

        if args.verbose:
            ok_info = f"{res['ok_f1']}/{res['ok_p']}/{res['ok_r']}"
            bad_info = f"{res['bad_f1']}/{res['bad_p']}/{res['bad_r']}"
            print(f'MCC: {mcc:.4f}\tOK(F1/P/R): {ok_info}\tBAD(F1/P/R): {bad_info}')

        ths += args.stride
        cnt += 1

    return opt_ths, max_mcc


def output_to_file(pred_fn, ths, output_fn):
    wf = open(output_fn, 'w')
    with open(pred_fn) as f:
        for line in f:
            line_probs = [float(f) for f in line.strip().split()]
            line_tags = ['OK' if p < ths else 'BAD' for p in line_probs]
            wf.write(' '.join(line_tags) + '\n')
    wf.close()


def main():
    args = parse_args()

    files_need_process = []
    suffix_need_process = []
    for suffix in ('source_tags', 'mtword_tags', 'gap_tags'):
        fn = args.pred_prefix + '.' + suffix + '.prob'
        if os.path.isfile(fn):
            files_need_process.append(fn)
            suffix_need_process.append(suffix)

    if len(files_need_process) > 0:
        print('You need to process prediction files:')
        for fn in files_need_process:
            print(fn)
        print('')
    else:
        raise FileNotFoundError(
            'Cannot find prediction files in prob form. '
            'Please make sure that some *.prob files exist in the path you specified.'
        )

    for i, pred_fn in enumerate(files_need_process):
        suffix = suffix_need_process[i]
        ref_fn = args.ref_prefix + '.' + suffix
        if not os.path.isfile(ref_fn):
            raise FileNotFoundError(f'Cannot find reference file: {ref_fn}')

        opt_ths, max_mcc = process(pred_fn, ref_fn, args)
        print(f'{suffix}: Optimized threshold [{opt_ths:.4f}], Maximum MCC [{max_mcc:.4f}]')

        if args.no_output:
            continue
        print('\nWriting optimized final results...')
        output_fn = args.pred_prefix + '.' + suffix
        output_to_file(pred_fn, opt_ths, output_fn)

    if args.no_output:
        return

    # generate  .tags
    mtword_res = args.pred_prefix + '.mtword_tags'
    gap_res = args.pred_prefix + '.gap_tags'

    if os.path.isfile(mtword_res) and os.path.isfile(gap_res):
        total_res = args.pred_prefix + '.tags'
        if args.use_predicted_gap:
            script_path = os.path.join(args.script_root, 'about-gap-tags', 'merge_mt_word_gap.py')
            cmd = f'python {script_path} -wt {mtword_res} -gt {gap_res} -o {total_res}'
        else:
            script_path = os.path.join(args.script_root, 'about-gap-tags', 'insert_ok_gaps.py')
            cmd = f'python {script_path} -i {mtword_res} -o {total_res}'

        os.system(cmd)

        print('\n\n')
        flag = input('Do you need to evaluate the final results? (y/n)')
        if flag == 'y':
            script_path = os.path.join(args.script_root, 'evaluation-scripts', 'word_evaluation.py')
            cmd = f'python {script_path} -r {args.ref_prefix} -p {args.pred_prefix}'
            os.system(cmd)


if __name__ == '__main__':
    main()
