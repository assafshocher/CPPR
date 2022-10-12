#!/bin/bash

# The SBATCH directives must appear before any executable line in this script.

#SBATCH --qos high2         # QOS (priority).
#SBATCH -N 1               # Number of nodes requested.
#SBATCH -n 1               # Number of tasks (i.e. processes).
#SBATCH --cpus-per-task=1  # Number of cores per task.
#SBATCH --gres=gpu:8       # Number of GPUs.
#SBATCH -t 60-00:00          # Time requested (D-HH:MM).
#SBATCH --nodelist=em6    # Uncomment if you need a specific machine.

# Uncomment this to have Slurm cd to a directory before running the script.
# You can also just run the script from the directory you want to be in.
#SBATCH -D /home/assafsho/mae_with_vqgan

# Uncomment to control the output files. By default stdout and stderr go to
# the same place, but if you use both commands below they'll be split up.
# %N is the hostname (if used, will create output(s) per node).
# %j is jobid.
##SBATCH -o slurm.%N.%j.out    # STDOUT
##SBATCH -e slurm.%N.%j.err    # STDERR

# Print some info for context.
source ~/.bashrc
conda activate taming
cd /home/assafsho/CPPR

nvidia-smi
# Python will buffer output of your script unless you set this.
# If you're not using python, figure out how to turn off output
# buffering when stdout is a file, or else when watching your output
# script you'll only get updated every several lines printed.
export PYTHONUNBUFFERED=1

OUTPUT_DIR='/home/assafsho/CPPR/cross_corr_gs_49_aug_0'
mkdir ${OUTPUT_DIR}
# DATA_PATH='/shared/group/ilsvrc'
DATA_PATH='/home/assafsho/data/ILSVRC2012'
TIME=$(date +%s%3N)
# Do all the research.
# python main_pretrain.py \
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
        --data_path ${DATA_PATH} \
        --model mae_vit_base_patch16 \
        --save_ckpt_freq 5 \
        --input_size 224 \
        --batch_size 512 \
        --warmup_epochs 40 \
        --epochs 800 \
        --group_sz 49 \
        --num_groups 2 \
        --blr 1.5e-4 \
        --temperature 0.1 \
        --output_dir ${OUTPUT_DIR}  \
        --project_name "linear_prob_comparison" \
        --dist_url "file://$OUTPUT_DIR/$TIME" \
        --weight_decay 0.5 \
        # --resume "/home/assafsho/mae/large/checkpoint-165.pth"  \
        

# Print completion time.
date