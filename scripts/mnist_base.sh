#!/bin/bash

#SBATCH -J 'fmnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=3
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-2

nwrong=(3000 3500 4000)

python sample_net.py --width 500 --depth 3 --lr 0.1 --tol 1e-4 \
         --num_wrong_samples ${nwrong[$SLURM_ARRAY_TASK_ID]} \
         --num_clean_samples 1000 \
         --n_iters 200000 \
         --batch_size -1 \
         --tol 5e-4

