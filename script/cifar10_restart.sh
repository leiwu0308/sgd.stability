#!/bin/bash

#SBATCH -J 'cifar10_collect'
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 1-20:00:00
#SBATCH --gres=gpu:1


lr=$1
bz=$2
DIR=experiments/cifar10/restart_full/lr${lr}_bz${bz}

python train.py --restart_file experiments/cifar/cifar10_teA64.58_lr1.00e-01_bz50000_momentum0.0_try1.pkl \
             --store_ckpt --store_last_ckpt --lr ${lr} --n_iters $3 \
             --batch_size ${bz} --iter_ckpt 40\
             --dir_ckpt ${DIR} \
             --dataset cifar10 --nclasses 10 --arch bigvgg \
             --num_clean_samples 50000 --loss cross_entropy

python analyze_traj.py --dir ${DIR} \
                --save_dir ${DIR} \
               --task loss \
			   --dataset cifar10 --arch bigvgg \
			   --num_clean_samples 50000 --nclasses 10 \
               --loss cross_entropy
