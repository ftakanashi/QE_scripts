#!/usr/bin/env python

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory",
                        help="A directory with subdirs containing different systems' outputs.")

    args = parser.parse_args()

    # validation
    assert os.path.isdir(args.directory), f"{args.directory} should be a valid directory."

    return args

def calc(reference, prediction):

    fpr, tpr, thresholds = metrics.roc_curve(reference, prediction)
    auc = metrics.auc(fpr, tpr)
    return fpr, tpr, auc

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

    plt.subplot(121)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    src_reference = flat_read_np(os.path.join(dir, "test.source_tags"), transfer=lambda x: 0.0 if x == 'OK' else 1.0)
    src_prob_fns = [os.path.join(dir, "Source", fn) for fn in os.listdir(os.path.join(dir, "Source")) if fn.endswith(".prob")]
    for fn in src_prob_fns:
        model_alias = os.path.basename(fn).replace(".prob", "")
        probs = flat_read_np(fn, transfer=lambda x: float(x))
        assert len(probs) == len(src_reference), f"{fn} tag number does not match reference"
        fpr, tpr, auc = calc(src_reference, probs)
        plt.plot(fpr, tpr, label=f"{model_alias} (AUC={auc:.4f})")
        plt.legend(loc='lower right', fontsize=8)

    plt.subplot(122)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    mt_reference = flat_read_np(os.path.join(dir, "test.mtword_tags"), transfer=lambda x: 0.0 if x == 'OK' else 1.0)
    mt_prob_fns = [os.path.join(dir, "MTWord", fn) for fn in os.listdir(os.path.join(dir, "MTWord")) if fn.endswith(".prob")]
    for fn in mt_prob_fns:
        model_alias = os.path.basename(fn).replace(".prob", "")
        probs = flat_read_np(fn, transfer=lambda x: float(x))
        assert len(probs) == len(mt_reference), f"{fn} tag number does not match reference"
        fpr, tpr, auc = calc(mt_reference, probs)
        plt.plot(fpr, tpr, label=f"{model_alias} (AUC={auc:.4f})")
        plt.legend(loc='lower right', fontsize=8)

    plt.show()

if __name__ == "__main__":
    main()