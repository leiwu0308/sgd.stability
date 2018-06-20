#!/bin/bash

#SBATCH -J 'fmnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=3
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-3

width=(110000 120000 130000 140000)

python sample_net.py --lr 0.1  \
          --batch_size 1200  --num_clean_samples 1000 --num_wrong_samples 200 --tol 1e-4 --n_iters 100000 \
          --iter_display 50  --momentum 0.9 --decay 80000 \
          --width ${width[$SLURM_ARRAY_TASK_ID]} --depth 1
