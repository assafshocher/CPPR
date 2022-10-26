#!/bin/bash

source ~/.bashrc

JOB_DIR=/checkpoint/amirbar/video_mae/logs_dir/
DATA_PATH=/datasets01/imagenet_full_size/061417
KILL_AFTER=80
COUNTER=0
partition=learnlab

#for BLR in 1.5e-4 1.5e-5
#do
#  for GROUP_SZ in 49
#  do
#    for COEFF_GINVAR in 25
#    do
#      for COEFF_BVAR in 0.
#      do
#        for COEFF_PVAR in 25
#        do
#          for COEFF_FCOV in 1e3 1e4 1e5
#          do
#            for COEFF_VAR_THR in 1.
#            do
#                OUTPUT_DIR="POSCROSS_FIXED_blr${BLR}_gsz${GROUP_SZ}_ginvar${COEFF_GINVAR}_bvar${COEFF_BVAR}_pvar${COEFF_PVAR}_fcov${COEFF_FCOV}_thr${COEFF_VAR_THR}_coeff_pcross_${COEFF_FCOV}"
#                python submitit_pretrain.py \
#                        --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
#                        --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
#                        --data_path ${DATA_PATH} \
#                        --nodes 1 \
#                        --use_volta32 \
#                        --batch_size 224 \
#                        --model mae_vit_base_patch16 \
#                        --save_ckpt_freq 5 \
#                        --input_size 224 \
#                        --warmup_epochs 40 \
#                        --epochs 1600 \
#                        --blr ${BLR} \
#                        --coeff_ginvar ${COEFF_GINVAR} \
#                        --coeff_bvar ${COEFF_BVAR} \
#                        --coeff_pvar ${COEFF_PVAR} \
#                        --coeff_fcov ${COEFF_FCOV} \
#                        --coeff_pcross ${COEFF_FCOV} \
#                        --coeff_var_thr ${COEFF_VAR_THR} \
#                        --weight_decay 0.05 \
#                        --output_dir ${OUTPUT_DIR}  \
#                        --project_name "linear_prob_comparison" \
#                        --dist_url "file://$OUTPUT_DIR/$TIME" \
#                        --kill_after ${KILL_AFTER} \
#                        --partition ${partition} \
#                        --use_volta32
#                COUNTER=$((COUNTER + 1))
#            done
#          done
#        done
#      done
#    done
#  done
#done

for BLR in 1.5e-4
do
  for GROUP_SZ in 49
  do
    for COEFF_GINVAR in 25
    do
      for COEFF_BVAR in 0.
      do
        for COEFF_PVAR in 25
        do
          for COEFF_FCOV in 1e3
          do
            for COEFF_VAR_THR in 1.
            do
                for coeff_pcross in 1e2 1e4 1e5
                do
                  OUTPUT_DIR="POSCROSS_FIXED_blr${BLR}_gsz${GROUP_SZ}_ginvar${COEFF_GINVAR}_bvar${COEFF_BVAR}_pvar${COEFF_PVAR}_fcov${COEFF_FCOV}_thr${COEFF_VAR_THR}_coeff_pcross_${coeff_pcross}"
                  python submitit_pretrain.py \
                          --job_dir ${JOB_DIR}/${OUTPUT_DIR} \
                          --output_dir ${JOB_DIR}/${OUTPUT_DIR}  \
                          --data_path ${DATA_PATH} \
                          --nodes 1 \
                          --use_volta32 \
                          --batch_size 224 \
                          --model mae_vit_base_patch16 \
                          --save_ckpt_freq 5 \
                          --input_size 224 \
                          --warmup_epochs 40 \
                          --epochs 1600 \
                          --blr ${BLR} \
                          --coeff_ginvar ${COEFF_GINVAR} \
                          --coeff_bvar ${COEFF_BVAR} \
                          --coeff_pvar ${COEFF_PVAR} \
                          --coeff_fcov ${COEFF_FCOV} \
                          --coeff_pcross ${coeff_pcross} \
                          --coeff_var_thr ${COEFF_VAR_THR} \
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
done

