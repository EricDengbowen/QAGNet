#!/usr/bin/bash

#SBATCH -J Evaluation_SIFR_ResNet50_Limited_False
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem 50G
#SBATCH -p amp20
#SBATCH -q amp20
#SBATCH -o ./slurmOutput/slurm-%j-%x.out
source /usr2/share/gpu.sbatch
python plain_test.py --num-gpus 1

