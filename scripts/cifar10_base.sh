#!/bin/bash

#SBATCH -J 'cifar10-vgg'
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -o "cifar10_n10000"

python sample_net.py --lr 0.01 --save_dir experiments/cifar10/ \
          --batch_size 100  --num_clean_samples 10000 --tol 1e-3 --n_iters 100000 \
          --dataset cifar10 --nclasses 10 --arch bigvgg --loss cross_entropy \
          --iter_display 500  --momentum 0.9 
