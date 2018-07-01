#!/bin/bash

#SBATCH -J 'eval'
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1

python analyze_net.py --model experiments/tmp/ --save_res fmnist_momentum.pkl --nonuniformity \
        --width 500 --depth 3 \
        --num_clean_samples 1000 \
        --num_wrong_samples 0 


