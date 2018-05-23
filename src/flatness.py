import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from .analysis import eval_accuracy


def random_perturb_net(net,ct,dataloader,stddev=1e-2,n_samples=100):
    net_new = deepcopy(net)
    paras_new = list(net_new.parameters())
    paras = list(net.parameters())

    L_old,_,_ = eval_accuracy(net,ct,dataloader)
    L_new = []
    for i in range(n_samples):
        for p_new,p in zip(paras_new,paras):
            p_new.data.normal_(0,stddev).add_(p.data)
        
        L,_,_ = eval_accuracy(net_new,ct,dataloader)
        L_new.append(L)

    return L_new,L_old


def compute_flatness(net,ct,dataloader,stddev=1e-2,n_samples=100):
    L_new,L_old = random_perturb_net(net, ct, dataloader,stddev,n_samples)
    L_new = np.mean(np.array(L_new))
    return (L_new-L_old)/stddev/stddev
