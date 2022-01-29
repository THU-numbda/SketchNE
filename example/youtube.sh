#!/bin/bash
set -x

EXE=../sketchne

if [ -z "$1" ]; then
    INPUT="../data/youtube.adj"
else
    INPUT=$1
fi

if [ -z "$2" ]; then
    NEOUT="youtube.emb"
    SPECOUT="youtube.spec"
else
    NEOUT="$2.emb"
    SPECOUT="$2.spec"
fi

if [ -z "$3" ]; then
	LABEL=../data/youtube.mat
else
	LABEL=$3
fi

[ ! -f $INPUT ] && python ../util/formatadj.py --file $LABEL --output $INPUT

(/usr/bin/time -v numactl -i all $EXE\
    -emb_out $NEOUT \
    -spec_out $SPECOUT \
    -s -m -rounds 1\
    -window_size 10 \
    -negative_samples 1\
    -gbbs_sparse_mm 1\
    -alpha 0.35\
    -convex_projection 0\
    -eig_rank 256\
    -power_iteration 30\
    -oversampling 20\
    -emb_dim 128\
    -eta1 8\
    -eta2 8\
    -s1 100\
    -s2 1000\
    -normalize 1\
    -step 10\
    -theta 0.5\
    -mu 0.2\
    -upper 0\
    -analyze 1\
    $INPUT ) |& tee -a youtube_ssr.log

python predict.py --label $LABEL --embedding $SPECOUT --seed 0 --C 2 --start-train-ratio 1 --stop-train-ratio 10 --num-train-ratio 10 --num-split 5

