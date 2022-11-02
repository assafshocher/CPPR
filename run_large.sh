#!/bin/bash

source ~/.bashrc

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir/
DATA_PATH=/datasets01/imagenet_full_size/061417
KILL_AFTER=801
partition=learnlab
batch_size=64
MASK_RATIO=0.75
BLR=1.5e-4
loss_invar_coeff=25
loss_var_coeff=25
loss_cov_coeff=767
OUTPUT_DIR="CONTEXTLESS_blr${BLR}_mr${MASK_RATIO}_invar${loss_invar_coeff}_var${loss_var_coeff}_cov${loss_cov_coeff}_batchsize_${batch_size}_large_8nodes"
python submitit_pretrain.py \
        --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
        --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
        --data_path ${DATA_PATH} \
        --nodes 8 \
        --batch_size ${batch_size} \
        --model mae_vit_large_patch16 \
        --save_ckpt_freq 5 \
        --input_size 224 \
        --warmup_epochs 40 \
        --epochs 1600 \
        --blr ${BLR} \
        --mask_ratio ${MASK_RATIO} \
        --loss_invar_coeff ${loss_invar_coeff} \
        --loss_var_coeff ${loss_var_coeff} \
        --loss_cov_coeff ${loss_cov_coeff} \
        --weight_decay 0.05 \
        --output_dir ${OUTPUT_DIR}  \
        --project_name "linear_prob_comparison" \
        --dist_url "file://$OUTPUT_DIR/$TIME" \
        --kill_after ${KILL_AFTER} \
        --partition ${partition} \
        --use_volta32