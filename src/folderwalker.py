import os
import time
import pickle
from math import sqrt 

import torch
import torch.nn as nn
from .analysis import weight_norm, eval_accuracy
from .linalg import eigen_hessian, eigen_variance
from .utils import get_train_info


class FolderWalker:
    def __init__(self,net,ct=None,max_iter=500000):
        self.net = net
        self.ct = ct
        self.max_iter = max_iter

    def run(self,dirname,dataloader,get_info,verbose=False):
        traj = []
        for filename in os.listdir(dirname):
            if filename.find('batchsize') == -1:
                continue
            iter_start,iter_now,iter_end,batch_size = get_train_info(filename)
            if iter_now > self.max_iter:
                continue

            self.net.load_state_dict(torch.load(
                    os.path.join(dirname,filename))
                )
            infos = get_info(self.net,self.ct,dataloader)

            traj.append({'iter':iter_now,'info':infos})
            if verbose:
                print('iter: %d finished'%(iter_now))

        traj = sorted(traj,key = lambda t:t['iter'])
        res = {'traj':traj}
        res['batch_size'] = batch_size

        return res

    def comp_norm(self,dirname):
        traj = []
        for filename in os.listdir(dirname):
            iter_start,iter_now,iter_end,batch_size = get_train_info(filename)
            self.net.load_state_dict(torch.load(
                    os.path.join(dirname,filename))
                )
            norm_now = compute_weight_norm(self.net)
            traj.append({'iter':iter_now,'norm':norm_now})
        traj = sorted(traj,key=lambda t:t['iter'])

        return traj



def scan(net,ct,trDL,teDL,dirname,
            niters=20,tol=1e-3,verbose=True,nonuniformity=True):
    count = 0
    res = []
    print(
        '--------------------------------------------------------------------------------------------------')
    print('     exp  time s lr\t batch_size | sharpnes\t diversity\t | loss    \t accuracy   weight_norm')
    print(
        '--------------------------------------------------------------------------------------------------')
    for filename in os.listdir(dirname):
        if not filename.endswith('pkl'):
            continue
        time_start = time.time()
        path_abs = os.path.join(dirname,filename)
        if os.path.isdir(path_abs):
            continue

        str = filename.split('_')
        teA = float(str[1][3:])
        lr = float(str[2][2:])
        batch_size = int(str[3][2:])
        momentum = float(str[4][8:])
        print(path_abs)
        print(net)

        net.load_state_dict(torch.load(path_abs))

        teL,teA,_ = eval_accuracy(net, ct, teDL)
        H_mu = eigen_hessian(net, ct, trDL, tol=tol, niters=niters)
        if nonuniformity:
            V_mu = eigen_variance(net, ct, trDL, tol=tol ,niters=niters)
        else: 
            V_mu = -1
        w_norm = weight_norm(net)

        res.append([lr,batch_size,momentum,H_mu,sqrt(V_mu),teL,teA,w_norm])
        if verbose:
            count = count + 1
            time_end = time.time()
            print('%3d-th, %.0f s\t %.1e\t %4d \t %.2f| %.2e\t %.2e\t | %.1e\t %.2f\t %.2e'%(
                count,time_end-time_start,lr,batch_size,momentum,H_mu,sqrt(V_mu),teL,teA,w_norm))
    return res
