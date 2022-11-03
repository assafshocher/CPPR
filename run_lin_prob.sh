#!/bin/bash

export PYTHONUNBUFFERED=1

PRETRAIN_CHKPT='/shared/assafsho/CONTEXTLESS_blr1.5e-4_mr0.75_invar25_var25_cov383_batchsize_64_basenorm_large_8nodes_checkpoint-100.pth'
OUTPUT_DIR='/home/assafsho/CPPR/tmp_lin_prob'

DATA_PATH='/home/assafsho/data/ILSVRC2012'

# Do all the research.
python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
        --batch_size 2048 \
        --model vit_large_patch16 \
        --finetune ${PRETRAIN_CHKPT} \
        --epochs 200 \
        --blr 0.01 \
        --weight_decay 0.0 \
        --dist_eval --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} \
        --cls_token \
        --resume '/shared/assafsho/amir/CONTEXTLESS_blr1.5e-4_mr0.75_invar25_var25_cov383_batchsize_64_basenorm_large_8nodes/checkpoint-100/checkpoint-20.pth'
# Print completion time.
date