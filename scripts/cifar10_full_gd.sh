#!/bin/bash

#SBATCH -J 'cifar10_collect'
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-20:00:00
#SBATCH --gres=gpu:1

python sample_net.py --lr 0.1 --save_dir experiments/cifar/ \
          --batch_size 50000  --num_clean_samples 50000 --tol 2e-3 --n_iters 5000 \
          --dataset cifar10 --nclasses 10 --arch bigvgg --loss cross_entropy \
          --iter_display 10  
