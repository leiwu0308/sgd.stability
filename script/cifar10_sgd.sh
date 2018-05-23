
ARCH=vgg
TRIES=$1

run(){
    python sample_net.py --dataset cifar10 --arch ${ARCH} --n_tries ${TRIES} \
        --lr 0.5 0.3 0.07 0.03 0.005 \
        --n_iters 50000 50000 100000 200000 400000\
        --tol 1e-4 \
        --batch_size $1 \
        --save_dir experiments/cifar10/sgd_collections \
        --gpuid $2
}

run 1000 0
run 400  0
run 100 0
run 10  0
run 4   0
