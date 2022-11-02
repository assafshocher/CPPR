#!/bin/bash

source ~/.bashrc

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir
DATA_PATH=/datasets01/imagenet_full_size/061417
MODEL_DIR=CONTEXTLESS_blr6.5625e-05_mr0.75_invar25_var25_cov767_batchsize_64_large_8nodes
COUNTER=0
for fn in checkpoint-100.pth checkpoint-200.pth checkpoint-300.pth checkpoint-400.pth checkpoint-500.pth
do
  if ((${COUNTER} > 1)); then
    partition=learnlab
  else
    partition=devlab
  fi
  PRETRAIN_CHKPT=${JOB_DIR}/${MODEL_DIR}/${fn}
  OUTPUT_DIR=${JOB_DIR}/${MODEL_DIR}/${fn%.pth}
  python submitit_linprobe.py \
      --accum_iter 4 \
      --job_dir ${OUTPUT_DIR} \
      --nodes 1 \
      --batch_size 512 \
      --model vit_large_patch16 --cls_token \
      --finetune ${PRETRAIN_CHKPT} \
      --epochs 50 \
      --blr 0.1 \
      --weight_decay 0.0 \
      --dist_eval --data_path ${DATA_PATH} \
      --partition ${partition}
  COUNTER=$((COUNTER + 1))
done