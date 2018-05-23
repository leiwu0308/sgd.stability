import os
import time
import sys
sys.path.insert(0,'..')
from copy import deepcopy
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from plot import parse_loss_traj
from src.data import load_mnist
from src.utils import load_model_net,process_config
from src.models.mnist import resfnn as fnn_mnist
from src.trainer import eval_accuracy


def train_one_epoch(net,ct,optimizer,imgs,labels,
                    iter_start=3000,display=False,max_iters=-1,cool=False):
    traj = []
    if max_iters < 0:
        max_iters = len(labels)
    net0 = deepcopy(net)
    for idx in range(max_iters):
        X = Variable(imgs[idx:idx+1].cuda())
        y = Variable(labels[idx:idx+1].cuda())

        if cool:
            net.load_state_dict(net0.state_dict())
        optimizer.zero_grad()
        logit = net(X)
        E = ct(logit,y)
        E.backward()
        optimizer.step()

        trL,trA,trC = eval_accuracy(net,trDL)
        teL,teA,teC = eval_accuracy(net,teDL)

        if display:
            print('%d, %d | \t %.2e  %.2f | \t %.2e  %.2f'%(
                idx, y.data[0], trL,trA,teL,teA
            ))
        traj.append((iter_start+idx,trL,trA,teL,teA,E.data[0]))

    traj = list(zip(*traj))
    return traj

def comp_wait_time(train_loss,threshold=1):
    s = np.asarray(train_loss)
    n = len(s)
    for i in range(n-1):
        if s[i] < 30:
            return i
    return n

if __name__ == '__main__':
    args = process_config('../configs/sgd_mnist.json')
    trDL,teDL = load_mnist(
                    root = '../data/fashionmnist',
                    fashion=True,
                    batch_size = 1200,
                    nsamples = args.num_clean_samples,
                    num_wrong = args.num_wrong_samples,
                    stop = True
                )
    net0 = fnn_mnist(depth=5).cuda()
    net0.load_state_dict(torch.load(
            os.path.join('../',args.restart_file)
    ))

    ct = torch.nn.CrossEntropyLoss()

    ntries = 2000
    wait_times = []
    print('try %d shuffles'%(ntries))
    for i in range(ntries):
        since = time.time()
        imgs,labels = trDL.X.clone(),trDL.y.clone()
        rnd_idx = torch.randperm(len(labels))
        imgs = imgs[rnd_idx]
        labels = labels[rnd_idx]
        net = deepcopy(net0)
        optimizer = torch.optim.SGD(net.parameters(),
                            lr = 0.01
                        )

        traj_new=train_one_epoch(net,ct,optimizer,imgs,labels,max_iters=1200)
        wait_times.append(comp_wait_time(traj_new[2]))
        if i%10==0:
            now = time.time()
            print('%d, take %.0f seconds'%(i,now-since))

    with open('wait_time.pkl','wb') as f:
        pickle.dump(wait_times,f)
