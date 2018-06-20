#!/bin/bash

#SBATCH -J 'eval'
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1

# python eval.py --model $1  --nonuniformity --width 500 --depth 3
python analyze_net.py --model $1 --save_res $2 --nonuniformity --width 500 --depth 3
