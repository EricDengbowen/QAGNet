#!/usr/bin/bash

#SBATCH -J QAGNet_trainRes50
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 50G
#SBATCH -q amp48
#SBATCH -p amp48
#SBATCH -o ./slurmOutput/slurm-%j-%x.out
source /usr2/share/gpu.sbatch
python train_net.py --num-gpus 1

