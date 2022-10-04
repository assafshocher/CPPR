#!/bin/bash

source ~/.bashrc

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir/
DATA_PATH=/datasets01/imagenet_full_size/061417
KILL_AFTER=30
COUNTER=0
for BLR in 1.5e-4 1e-5 1e-4 1e-3
do
    for TEMPERATURE in 0.05 0.1 0.5 1.0
    do
        if ((${COUNTER} > 3)); then
          partition=learnfair
        else
          partition=devlab
        fi

        OUTPUT_DIR="cmae_temp_${TEMPERATURE}_blr_${BLR}"
        python submitit_pretrain.py \
                --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
                --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                --data_path ${DATA_PATH} \
                --temperature ${TEMPERATURE} \
                --nodes 1 \
                --use_volta32 \
                --batch_size 128 \
                --model mae_vit_base_patch16 \
                --save_ckpt_freq 5 \
                --input_size 224 \
                --warmup_epochs 40 \
                --epochs 800 \
                --blr ${BLR} \
                --weight_decay 0.05 \
                --output_dir ${OUTPUT_DIR}  \
                --project_name "linear_prob_comparison" \
                --dist_url "file://$OUTPUT_DIR/$TIME" \
                --kill_after ${KILL_AFTER} \
                --partition ${partition} \
                --use_volta32
        COUNTER=$((COUNTER + 1))
    done
done
