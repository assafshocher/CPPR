#!/bin/bash

source ~/.bashrc

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir/
DATA_PATH=/datasets01/imagenet_full_size/061417
KILL_AFTER=80
COUNTER=0

for DETACH in '' '--detach'
do
  for W_PRED_LOSS in 1.0 0.5 0.1
  do
    for W_BATCHWISE_LOSS in 1.0
    do
      for W_PATCHWISE_LOSS in 1.0
      do
        for BLR in 1.5e-4
        do
          for TEMPERATURE in 0.1 1.0
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
                      --w_pred_loss ${W_PRED_LOSS} \
                      --w_batchwise_loss ${W_BATCHWISE_LOSS} \
                      --w_patchwise_loss ${W_PATCHWISE_LOSS} \
                      ${DETACH} \
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
      done
    done
  done
done
