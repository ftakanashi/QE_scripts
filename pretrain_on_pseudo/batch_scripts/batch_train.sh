#!/bin/bash

for i in {0..9}
do
  rm -rf data.$i/models
  mkdir -p data.$i/models
  python fairseq/train.py data.$i/data-bin -a transformer --optimizer adam --lr 2e-5 -s en -t zh --label-smoothing 0.1 --dropout 0.3 --max-tokens 4096 --min-lr 1e-9 --lr-scheduler inverse_sqrt --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --keep-last-epochs 5 --max-update 100000 --warmup-updates 4000 --warmup-init-lr 1e-7 --adam-betas '(0.9,0.98)' --save-dir data.$i/models --no-progress-bar > data.$i/models/train.log 2>&1
done