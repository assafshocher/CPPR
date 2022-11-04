#!/bin/bash

# Print some info for context.
source ~/.bashrc
conda activate taming
cd /home/assafsho/CPPR
nvidia-smi
export PYTHONUNBUFFERED=1
JOB_DIR='/home/assafsho/CPPR/'
mkdir ${JOB_DIR}/${OUTPUT_DIR} 
DATA_PATH='/home/assafsho/data/ILSVRC2012'
TIME=$(date +%s%3N)
KILL_AFTER=8000
batch_size=512

for contextless_model in 'base_norm'
do
  for BLR in 1.5e-4
  do
    for MASK_RATIO in 0.75
    do
      for loss_invar_coeff in 25
      do
        for loss_var_coeff in 25
        do
          for loss_cov_coeff in 383
          do
            OUTPUT_DIR="masked_eval_CONTEXTLESS_${contextless_model}_blr${BLR}_mr${MASK_RATIO}_invar${loss_invar_coeff}_var${loss_var_coeff}_cov${loss_cov_coeff}_batchsize_${batch_size}"
            python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
                    --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                    --data_path ${DATA_PATH} \
                    --batch_size ${batch_size} \
                    --model mae_vit_base_patch16 \
                    --contextless_model ${contextless_model} \
                    --save_ckpt_freq 5 \
                    --input_size 224 \
                    --use_batch_stats\
                    --warmup_epochs 40 \
                    --epochs 1600 \
                    --blr ${BLR} \
                    --mask_ratio ${MASK_RATIO} \
                    --mask_ratio_eval ${MASK_RATIO} \
                    --loss_invar_coeff ${loss_invar_coeff} \
                    --loss_var_coeff ${loss_var_coeff} \
                    --loss_cov_coeff ${loss_cov_coeff} \
                    --weight_decay 0.05 \
                    --project_name "linear_prob_comparison" \
                    --kill_after ${KILL_AFTER} 
          done
        done
      done
    done
  done
done


# -m torch.distributed.launch --nproc_per_node=8