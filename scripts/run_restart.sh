#python sample_net.py --lr 0.1 --n_iters 20000 --save_dir experiments/corrupt_data/ --gpuid 1 --batch_size 1200 --num_wrong_samples 200

# python train.py --restart_file experiments/corrupt_data/fashionmnist_teA71.61_lr1.00e-01_bz1200_momentum0.0_try1.pkl \
#             --store_ckpt --store_last_ckpt --lr 0.1 --n_iters 5000 \
#             --num_wrong_samples 200 --batch_size 4 --iter_ckpt 40\
#             --dir_ckpt experiments/corrupt_data/lr0.3_bz1200

python analyze_traj.py --dir experiments/corrupt_data/lr0.1_bz4 \
                --save_dir experiments/corrupt_data/lr0.1_bz4/ \
               --task loss

python analyze_traj.py --dir experiments/corrupt_data/lr0.1_bz4 \
                --save_dir experiments/corrupt_data/lr0.1_bz4/ \
               --task sharpness
