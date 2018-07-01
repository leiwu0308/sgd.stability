#!/bin/bash

#SBATCH -J 'cifar10'
#SBATCH -N 1
#SBATCH --cpus-per-task=3
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-6

momentum=(0.0 0.2 0.4 0.6 0.8 0.9 0.99)

python sample_net.py --arch vgg --lr 0.01 --tol 1e-4 \
         --dataset cifar10 \
         --num_clean_samples 1000 \
         --num_wrong_samples 0 \
         --n_tries 4 \
         --n_iters 1000000 \
         --batch_size 10 \
         --tol 1e-4 \
         --momentum ${momentum[$SLURM_ARRAY_TASK_ID]} \
         --save_dir experiments/cifar10 \
         --nclasses 2

