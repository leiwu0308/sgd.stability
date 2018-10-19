#!/bin/bash

#SBATCH -J 'eval'
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH -t 20:00:00
#SBATCH --gres=gpu:1
#SBATCH -a 0-3

ARCH=resnet
TRIES=2
batchsize=(4 10 100 1000)

run(){
    python sample_net.py --dataset cifar10 --arch ${ARCH} --n_tries ${TRIES} \
        --lr 0.01 0.05 \
        --n_iters 50000 50000 \
        --tol 5e-4 \
        --batch_size $1 \
        --save_dir experiments/cifar10/scatter \
        --nclasses 2 \
        --num_clean_samples 500 
}

run ${batchsize[$SLURM_ARRAY_TASK_ID]}
