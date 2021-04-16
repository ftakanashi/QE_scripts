#!/usr/bin/env python

import os

def main():
    threshold = 0.05
    cmd = f'python ~/QE_scripts/source-mt-align/pred_to_align_with_tag.py ' \
          f'-p predictions_.json -ao src_mt_dev.joint_pred.align -sto pred.source_tags ' \
          f'-tto pred.mtword_tags --tag_prob_threshold 0.45'