#!/usr/bin/env python

'''
This script performs significant test and draw ROC curve along with AUC value by
reading in labels and predictions from different models of Source tags and MT Word tags.

The structure of --directory should be like：
├── MTWord    # [fixed name] Directory for probability files of MT word tags of different systems
│   ├── baseline.prob    # prob files must ends with .prob
│   ├── model1.prob
│   └── model2.prob
├── Source    # [fixed name] Directory for probability files of source tags of different systems
│   ├── baseline.prob    # prob files must ends with .prob
│   ├── model1.prob
│   └── model2.prob
├── test.mtword_tags    # [fixed name] Labels of MT word tags
└── test.source_tags    # [fixed name] Labels of source tags
'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics
from sklearn.metrics import roc_auc_score

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory",
                        help="A directory with subdirs containing different systems' outputs.")

    parser.add_argument("--do_significant_test", action="store_true", default=False,
                        help="Set the flag to do significant test.")
    parser.add_argument("--baseline_alias", type=str, default='OpenKiwi',
                        help="Alias of the baseline system.")

    args = parser.parse_args()

    # validation
    assert os.path.isdir(args.directory), f"{args.directory} should be a valid directory."

    return args

def calc(reference, prediction):

    fpr, tpr, thresholds = metrics.roc_curve(reference, prediction)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc

def permutation_test(ref, probs1, probs2, nsamples=1000):
    '''
    https://stats.stackexchange.com/questions/214687/what-statistical-tests-to-compare-two-aucs-from-two-models-on-the-same-dataset
    https://stackoverflow.com/questions/52373318/how-to-compare-roc-auc-scores-of-different-binary-classifiers-and-assess-statist
    assuming probs2 are predicted by the baseline and
    probs1 are predicted by the system to be tested
    '''
    orig_auc = roc_auc_score(ref, probs1)
    cnt = 0
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(ref))
        p = np.where(mask, probs1, probs2)
        auc_p = roc_auc_score(ref, p)
        if auc_p >= orig_auc:
            cnt += 1
    p_val = cnt / nsamples
    return p_val

def main():

    def flat_read_np(fn, transfer=None):
        with open(fn) as f:
            res = []
            for l in f:
                for ele in l.strip().split():
                    if transfer is not None:
                        ele = transfer(ele)
                    res.append(ele)
            return np.array(res)

    args = parse_args()
    dir = args.directory

    plt.figure(figsize=(6.4, 2.4))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    # Source Tag
    plt.subplot(121)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    src_reference = flat_read_np(os.path.join(dir, "test.source_tags"), transfer=lambda x: 0.0 if x == 'OK' else 1.0)
    src_prob_fns = [os.path.join(dir, "Source", fn) for fn in os.listdir(os.path.join(dir, "Source")) if fn.endswith(".prob")]
    baseline_probs = flat_read_np(os.path.join(dir, "Source", f"{args.baseline_alias}.prob"), transfer=lambda x: float(x))
    for fn in src_prob_fns:
        model_alias = os.path.basename(fn).replace(".prob", "")
        probs = flat_read_np(fn, transfer=lambda x: float(x))
        assert len(probs) == len(src_reference), f"{fn} tag number does not match reference"

        fpr, tpr, auc = calc(src_reference, probs)
        label = f"{model_alias} (AUC={auc:.3f})"

        if args.do_significant_test and args.baseline_alias != model_alias:
            # do significant test (permutation test) against baseline
            p = permutation_test(src_reference, probs, baseline_probs)
            print(f"[Source]P-value of {model_alias}: {p}")
            if 0.01 < p <= 0.05:
                label = f"{model_alias}~* (AUC={auc:.3f})"
            elif p <= 0.01:
                label = f"{model_alias}* (AUC={auc:.3f})"

        plt.plot(fpr, tpr, label=label)
        plt.legend(loc='lower right', fontsize=8)

    # MT Tag
    plt.subplot(122)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    mt_reference = flat_read_np(os.path.join(dir, "test.mtword_tags"), transfer=lambda x: 0.0 if x == 'OK' else 1.0)
    mt_prob_fns = [os.path.join(dir, "MTWord", fn) for fn in os.listdir(os.path.join(dir, "MTWord")) if fn.endswith(".prob")]
    baseline_probs = flat_read_np(os.path.join(dir, "MTWord", f"{args.baseline_alias}.prob"), transfer=lambda x: float(x))
    for fn in mt_prob_fns:
        model_alias = os.path.basename(fn).replace(".prob", "")
        probs = flat_read_np(fn, transfer=lambda x: float(x))
        assert len(probs) == len(mt_reference), f"{fn} tag number does not match reference"

        fpr, tpr, auc = calc(mt_reference, probs)
        label = f"{model_alias} (AUC={auc:.3f})"

        if args.do_significant_test and args.baseline_alias != model_alias:
            # do significant test (permutation test) against baseline
            p = permutation_test(mt_reference, probs, baseline_probs)
            print(f"[MT]P-value of {model_alias}: {p}")
            if 0.01 < p <= 0.05:
                label = f"{model_alias}~* (AUC={auc:.3f})"
            elif p <= 0.01:
                label = f"{model_alias}* (AUC={auc:.3f})"


        plt.plot(fpr, tpr, label=label)
        plt.legend(loc='lower right', fontsize=8)

    plt.show()

if __name__ == "__main__":
    main()