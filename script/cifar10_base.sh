#!/bin/bash

python sample_net.py --lr 0.01 --save_dir experiment/cifar10/ \
          --batch_size 100  --num_clean_samples 50000 --tol 1e-3 --n_iters 50000 \
          --dataset cifar10 --nclasses 10 --arch bigvgg --loss cross_entropy \
          --iter_display 500  --momentum 0.9 
