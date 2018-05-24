#! /bin/bash

#SBATCH -J 'calc_sharpness'
#SBATCH -N 1
#SBATCH --cpus-per-task=5
#SBATCH -t 1-20:00:00
#SBATCH --gres=gpu:1

module avail
module load cudann/cuda-8.0/5.1
python analyze_net.py --dataset cifar10 --arch vgg --save $1 --dir $2 --nonuniformity
