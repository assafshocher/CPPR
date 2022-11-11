#!/bin/bash

source ~/.bashrc
conda activate fairvit

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir/
DATA_PATH=/datasets01/imagenet_full_size/061417
KILL_AFTER=200
COUNTER=0
batch_size=224
partition=learnlab
contextless_model_projector_arch=768-768
for BLR in 1.5e-4
do
  for MASK_RATIO in 0.75
  do
    for loss_invar_coeff in 25
    do
      for loss_var_coeff in 50 100
      do
        for loss_cov_coeff in 383 767 1534
        do
          for model_projector_arch in 768-768-768
          do
            for norm_vit_projector in 1
            do
              OUTPUT_DIR="CONTEXTLESS_blr${BLR}_mr${MASK_RATIO}_invar${loss_invar_coeff}_var${loss_var_coeff}_cov${loss_cov_coeff}_batchsize_${batch_size}_custombasenorm_${contextless_model_projector_arch}_modelproj_${model_projector_arch}_norm_vit_projector_${norm_vit_projector}_fixed_diag"
              python submitit_pretrain.py \
                      --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
                      --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                      --data_path ${DATA_PATH} \
                      --nodes 1 \
                      --batch_size ${batch_size} \
                      --model mae_vit_base_patch16 \
                      --save_ckpt_freq 5 \
                      --input_size 224 \
                      --warmup_epochs 40 \
                      --epochs 1600 \
                      --blr ${BLR} \
                      --mask_ratio ${MASK_RATIO} \
                      --contextless_model custom_base_norm \
                      --contextless_model_projector_arch ${contextless_model_projector_arch} \
                      --model_projector_arch ${model_projector_arch} \
                      --loss_invar_coeff ${loss_invar_coeff} \
                      --loss_var_coeff ${loss_var_coeff} \
                      --loss_cov_coeff ${loss_cov_coeff} \
                      --norm_vit_projector ${norm_vit_projector} \
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

