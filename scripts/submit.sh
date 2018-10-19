#!/bin/bash

#SBATCH -J 'eval_test'
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1

# python eval.py --model $1  --nonuniformity --width 500 --depth 3

# python analyze_net.py --model $1 --save_res $2 --nonuniformity \
#         --width 500 --depth 3 \
#         --num_clean_samples 1000 \
#         --num_wrong_samples 0 


python analyze_net.py --model $1 --save_res $2 --nonuniformity \
             --dataset cifar10 \
             --num_clean_samples 5000 \
             --num_wrong_samples 0 \
             --nclasses 2 \
             --arch resnet
