#!/bin/bash

source ~/.bashrc

JOB_DIR=
DATA_PATH=
KILL_AFTER=30

for BLR in 1.5e-4 1e-5 1e-4 1e-3
do
    for TEMPERATURE in 0.05 0.1 0.5 1.0
    do
        OUTPUT_DIR="cmae_temp_${TEMPERATURE}_blr_${BLR}"
        python submitit_pretrain.py \
                --job_dir ${JOB_DIR} \
                --output_dir ${OUTPUT_DIR}  \
                --data_path ${DATA_PATH} \
                --temperature temperature \
                --nodes 1 \
                --use_volta32 \
                --batch_size 256 \
                --model mae_vit_base_patch16 \
                --save_ckpt_freq 5 \
                --input_size 224 \
                --warmup_epochs 40 \
                --epochs 800 \
                --blr BLR \
                --weight_decay 0.05 \
                --temperature TEMPERATURE \
                --output_dir ${OUTPUT_DIR}  \
                --project_name "linear_prob_comparison" \
                --dist_url "file://$OUTPUT_DIR/$TIME" \
                --kill_after KILL_AFTER \
                # --resume "checkpoint-165.pth"  \
    done
done
