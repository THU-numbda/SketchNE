#!/bin/bash
set -x

EXE=../sketchne
LOG=log_mag

if [ -z "$1" ]; then
    INPUT="../data/mag.adj"
else
    INPUT=$1
fi

if [ -z "$2" ]; then
    NEOUT="mag.emb"
    SPECOUT="mag.spec"
else
    NEOUT="$2.emb"
    SPECOUT="$2.spec"
fi

if [ -z "$2" ]; then
	LABEL=../data/mag.label.npz
else
	LABEL=$2
fi

mkdir -p $LOG
NOW=$(date +"%Y-%m-%d")

[ ! -f $INPUT ] && python ../util/formatadj.py --file ../data/mag.edge --output $INPUT

(/usr/bin/time -v numactl -i all $EXE\
   -emb_out $NEOUT \
   -spec_out $SPECOUT \
   -s -m -rounds 1\
   -window_size 10\
   -negative_samples 1\
   -gbbs_sparse_mm 1\
   -alpha 0.45\
   -convex_projection 0\
   -eig_rank 256\
   -power_iteration 30\
   -oversampling 20\
   -emb_dim 128\
   -eta1 8\
   -eta2 8\
   -s2 100\
   -s3 1000\
   -normalize 1\
   -step 10\
   -theta 0.5\
   -mu 0.2\
   -upper 0\
   -analyze 1\
   $INPUT) |& tee -a $LOG/$NOW.log


(python predict.py --label $LABEL --embedding $SPECOUT --seed 0 --C 10 --start-train-ratio 0.001 --stop-train-ratio 0.01  --num-train-ratio 2 --num-split 5)|& tee -a $LOG/$NOW.log
(python predict.py --label $LABEL --embedding $SPECOUT --seed 0 --C 10 --start-train-ratio 0.1 --stop-train-ratio 1  --num-train-ratio 2 --num-split 5)|& tee -a $LOG/$NOW.log