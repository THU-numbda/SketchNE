#!/bin/bash

set -x

EXE=../sketchne

if [ -z "$1" ]; then
    INPUT="../data/blog.adj"
else
    INPUT=$1
fi

if [ -z "$2" ]; then
    NEOUT="blog.emb"
    SPECOUT="blog.spec"
else
    NEOUT="$2.emb"
    SPECOUT="$2.spec"
fi

if [ -z "$3" ]; then
	LABEL=../data/blogcatalog.mat
else
	LABEL=$2
fi

[ ! -f $INPUT ] && python ../util/formatadj.py --file $LABEL --output $INPUT

(/usr/bin/time -v numactl -i all $EXE\
    -emb_out $NEOUT \
    -spec_out $SPECOUT \
    -s -m -rounds 1\
    -window_size 10 \
    -negative_samples 1\
    -gbbs_sparse_mm 1\
    -alpha 0.5\
    -convex_projection 0\
    -eig_rank 256\
    -power_iteration 20\
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
    $INPUT ) |& tee -a blog_ssr.log

python predict.py --label $LABEL --embedding $SPECOUT --seed 0 --C 3 --start-train-ratio 10 --stop-train-ratio 90 --num-train-ratio 9 --num-split 5
