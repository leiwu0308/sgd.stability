gain=1

# nohup python sample_net.py --lr 0.1 --n_iters 30000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 1200 --num_wrong_samples 200 --n_tries 5  --gain ${gain} &

# nohup python sample_net.py --lr 0.5 --n_iters 20000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 1200 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

# wait 
# nohup python sample_net.py --lr 0.05 --n_iters 50000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 1200 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

# nohup python sample_net.py --lr 1 --n_iters 10000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 1200 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

# nohup python sample_net.py --lr 2 --n_iters 10000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 1200 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

#######################################
wait 
nohup python sample_net.py --lr 0.1 --n_iters 30000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 400 --num_wrong_samples 200 --n_tries 5 --gain ${gain}  &

nohup python sample_net.py --lr 0.5 --n_iters 40000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 400 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

wait 
nohup python sample_net.py --lr 0.05 --n_iters 80000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 400 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

nohup python sample_net.py --lr 1 --n_iters 10000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 400 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

nohup python sample_net.py --lr 2 --n_iters 10000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 400 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &
#######################################

# wait 
# nohup python sample_net.py --lr 0.1 --n_iters 40000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 100 --num_wrong_samples 200 --n_tries 5 --gain ${gain}  &

# nohup python sample_net.py --lr 0.5 --n_iters 40000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 100 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

# wait 
# nohup python sample_net.py --lr 0.05 --n_iters 80000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 100 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

# nohup python sample_net.py --lr 1 --n_iters 10000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 100 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

# nohup python sample_net.py --lr 2 --n_iters 10000 \
#             --save_dir experiments/corrupt_data/ \
#           --batch_size 100 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &
#######################################


wait 
nohup python sample_net.py --lr 0.1 --n_iters 100000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 10 --num_wrong_samples 200 --n_tries 5 --gain ${gain}  &

nohup python sample_net.py --lr 0.5 --n_iters 100000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 10 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

wait 
nohup python sample_net.py --lr 0.05 --n_iters 100000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 10 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

nohup python sample_net.py --lr 1 --n_iters 50000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 10 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &

nohup python sample_net.py --lr 2 --n_iters 50000 \
            --save_dir experiments/corrupt_data/ \
          --batch_size 10 --num_wrong_samples 200 --n_tries 5 --gain ${gain} &


