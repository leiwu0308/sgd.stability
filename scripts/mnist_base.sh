#!/bin/bash

#SBATCH -J 'fmnist-fnn'
#SBATCH -N 1
#SBATCH --cpus-per-task=3
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-6

# nwrong=(3000 3500 4000)

# python sample_net.py --width 500 --depth 3 --lr 0.1 --tol 1e-4 \
#          --num_wrong_samples ${nwrong[$SLURM_ARRAY_TASK_ID]} \
#          --num_clean_samples 1000 \
#          --n_iters 200000 \
#          --batch_size -1 \
#          --tol 5e-4

momentum=(0.0 0.2 0.4 0.6 0.8 0.9 0.99)

python sample_net.py --width 500 --depth 3 --lr 0.1 --tol 1e-4 \
         --num_clean_samples 1000 \
         --num_wrong_samples 0 \
         --n_tries 4 \
         --n_iters 200000 \
         --batch_size 10 \
         --tol 1e-4 \
         --momentum ${momentum[$SLURM_ARRAY_TASK_ID]} \
         --save_dir experiments/tmp 

