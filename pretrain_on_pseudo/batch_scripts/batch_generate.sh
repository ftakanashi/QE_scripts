#!/bin/bash

for i in {0..9}
do
  rm -f data.$i/generate.log
  python fairseq/generate.py data.$i/data-bin --path data.$i/models/checkpoint_best.pt --max-tokens 8192 > data.$i/generate.log
done