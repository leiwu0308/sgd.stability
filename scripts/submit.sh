#!/bin/bash

#SBATCH -J 'cifar10_collect'
#SBATCH -N 1
#SBATCH --cpus-per-task=5
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1

module avail
module load cudann/cuda-8.0/5.1

python analyze_traj.py --dir $1 \
                --save_dir $1 \
               --task loss \
               --num_wrong_samples 200 

python analyze_traj.py --dir $1 \
                --save_dir $1 \
               --task sharpness \
               --num_wrong_samples 200
