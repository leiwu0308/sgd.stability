#!/bin/bash

#SBATCH -J 'eval'
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1


python analyze_net.py --model experiments/cifar10/scatter \
             --save_res cifar10_resnet_scatter.pkl \
             --nonuniformity \
             --dataset cifar10 \
             --num_clean_samples 1000 \
             --num_wrong_samples 0 \
             --nclasses 2 \
             --arch resnet
