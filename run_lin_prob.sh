#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos low         # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=1  # Number of cores per task.
#SBATCH --gres=gpu:1       # Number of GPUs.
#SBATCH -t 5-00:00          # Time requested (D-HH:MM).
#SBATCH --nodelist=em7    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
#SBATCH -D /home/yossi_gandelsman/code/deep_transformer_prior/mae_with_vqgan

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.
##SBATCH -o slurm.%N.%j.out    # STDOUT
##SBATCH -e slurm.%N.%j.err    # STDERR

# Print some info for context.
source ~/.bashrc
conda activate taming
cd /home/assafsho/mae

# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

PRETRAIN_CHKPT='/home/assafsho/mae/only_batchwise/checkpoint-799.pth'
OUTPUT_DIR='/home/assafsho/mae/base_lin_prob'

DATA_PATH='/home/assafsho/data/ILSVRC2012'

# Do all the research.
python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
        --batch_size 2048 \
        --model vit_base_patch16\
        --finetune ${PRETRAIN_CHKPT} \
        --epochs 90 \
        --blr 0.1 \
        --weight_decay 0.0 \
        --dist_eval --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} \
        # --cls_token \
# Print completion time.
date