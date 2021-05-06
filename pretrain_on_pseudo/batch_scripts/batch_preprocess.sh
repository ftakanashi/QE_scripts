#!/bin/bash

for i in {0..9}
do
  rm -rf data.$i/data-bin
  mkdir -p data.$i/data-bin
  python fairseq/preprocess.py -s en -t zh --trainpref data.$i/train --validpref data.$i/test --testpref data.$i/test --destdir data.$i/data-bin --srcdict dict.en.txt --tgtdict dict.zh.txt --workers 32 > data.$i/data-bin/preprocess.log 2>&1
done