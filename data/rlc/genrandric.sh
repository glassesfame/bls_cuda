#!/bin/bash
fromcopy="/home/sliu/digitalliance/data/rlc"
tocopy="/home/sliu/writeup-dec2025/bls_cuda/rlc"
while read ric; do
    fromric="${fromcopy}/${ric}/raw.h5"
    toric="${tocopy}/${ric}raw.h5"
    cp $fromric $toric
done < rand42ric.txt