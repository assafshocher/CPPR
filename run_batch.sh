#!/bin/bash

source ~/.bashrc

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir/
DATA_PATH=/datasets01/imagenet_full_size/061417
KILL_AFTER=120
COUNTER=0





for TEMPERATURE in 0.1
do
  for W_PRED_LOSS in 0.3333
  do
    for BLR in 1.5e-5
    do
      for W_BATCHWISE_LOSS in 0.33333
      do
        for W_PATCHWISE_LOSS in 0.3333
        do
          for GROUP_SZ in 49
          do
            for DETACH in '--detach'
            do
                if ((${COUNTER} > 3)); then
                  partition=learnfair
                else
                  partition=devlab
                fi

                OUTPUT_DIR="CROSS_tmp${TEMPERATURE}_blr${BLR}_gsz${GROUP_SZ}_wpr${W_PRED_LOSS}_wba${W_BATCHWISE_LOSS}_wpa${W_PATCHWISE_LOSS}${DETACH}"
                python submitit_pretrain.py \
                        --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
                        --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                        --data_path ${DATA_PATH} \
                        --temperature ${TEMPERATURE} \
                        --nodes 1 \
                        --use_volta32 \
                        --batch_size 224 \
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
done



for TEMPERATURE in 0.1
do
  for W_PRED_LOSS in 0.3333
  do
    for BLR in 1.5e-5
    do
      for W_BATCHWISE_LOSS in 0.33333
      do
        for W_PATCHWISE_LOSS in 0.3333
        do
          for GROUP_SZ in 49 20 10
          do
            for DETACH in ''
            do
                if ((${COUNTER} > 3)); then
                  partition=learnfair
                else
                  partition=devlab
                fi

                OUTPUT_DIR="CROSS_tmp${TEMPERATURE}_blr${BLR}_gsz${GROUP_SZ}_wpr${W_PRED_LOSS}_wba${W_BATCHWISE_LOSS}_wpa${W_PATCHWISE_LOSS}${DETACH}"
                python submitit_pretrain.py \
                        --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
                        --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                        --data_path ${DATA_PATH} \
                        --temperature ${TEMPERATURE} \
                        --nodes 1 \
                        --use_volta32 \
                        --batch_size 224 \
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
done




for TEMPERATURE in 0.1 1.0 
do
  for W_PRED_LOSS in 0.3333 0.1 1.0 3.333333
  do
    for BLR in 1.5e-5 1e-5 1.5e-4
    do
      for W_BATCHWISE_LOSS in 0.33333
      do
        for W_PATCHWISE_LOSS in 0.3333
        do
          for GROUP_SZ in 49
          do
            for DETACH in '' '--detach'
            do
                if ((${COUNTER} > 3)); then
                  partition=learnfair
                else
                  partition=devlab
                fi

                OUTPUT_DIR="CROSS_tmp${TEMPERATURE}_blr${BLR}_gsz${GROUP_SZ}_wpr${W_PRED_LOSS}_wba${W_BATCHWISE_LOSS}_wpa${W_PATCHWISE_LOSS}${DETACH}"
                python submitit_pretrain.py \
                        --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
                        --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                        --data_path ${DATA_PATH} \
                        --temperature ${TEMPERATURE} \
                        --nodes 1 \
                        --use_volta32 \
                        --batch_size 224 \
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
done
