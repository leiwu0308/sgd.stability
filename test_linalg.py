import os
import time
import torch
import argparse
from copy import deepcopy
from bunch import Bunch
# torch.manual_seed(20)

from src.utils import load_model, load_data
from src.net_utils import num_parameters
from src.trainer import Trainer
from src.analysis import eval_accuracy, weight_norm
from src.linalg import *

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('--dataset',default='fashionmnist')
argparser.add_argument('--model', default='experiments/')
argparser.add_argument('--gpuid',default='0,')
argparser.add_argument('--arch',default='fnn')
argparser.add_argument('--loss', default='mse')
argparser.add_argument('--num_wrong_samples',type=int,default=0)
argparser.add_argument('--num_clean_samples',type=int,default=1000)
argparser.add_argument('--nonuniformity', action='store_true')
args = argparser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

# Update parameters
args_fashionmnist = {
    'num_clean_samples': 1000,
    'num_wrong_samples': 0,
    'arch': 'fnn',
    'load_size': 2000,
    'dataset': 'fashionmnist',
    'batch_size':1000
}

args_cifar = {
    'num_clean_samples': 1000,
    'num_wrong_samples': 0,
    'arch': 'resnet',
    'load_size': 200,
    'dataset': 'cifar10',
    'batch_size': 200
}
if args.dataset == 'fashionmnist':
    config = args_fashionmnist
else:
    config = args_cifar
config.update(vars(args))
args = Bunch(config)

if args.loss == 'mse':
    ct = torch.nn.MSELoss().cuda()
    one_hot = True
elif args.loss == 'hinge':
    ct = torch.nn.MultiMarginLoss(p=2)
    one_hot = False
elif args.loss == 'cross_entropy':
    ct = torch.nn.CrossEntropyLoss()
    one_hot = False

#####################################
# Process
####################################
trDL,teDL = load_data(args,stop=True,one_hot=one_hot)
net = load_model(args.dataset,args.arch)
net.load_state_dict(torch.load(args.model))

# Evaluation
trL,trA,trC = eval_accuracy(net,ct,trDL)
teL,teA,teC = eval_accuracy(net,ct,teDL)
print('===> solution: ')
print('\t train loss: %.2e, acc: %.2f'%(trL,trA))
print('\t test loss: %.2e, acc: %.2f'%(teL,teA))
print('l2 norm %.2e'%(weight_norm(net)))

time_start = time.time()
mu = eigen_hessian(net,ct,trDL,verbose=True,tol=1e-5,niters=20)
time_end = time.time()
print('takes %.0f seconds'%(time_end-time_start))

if args.nonuniformity:
    time_start = time.time()
    mu = eigen_variance(net,ct,trDL,verbose=True,tol=1e-5,niters=20)
    time_end = time.time()
    print('takes %.0f seconds'%(time_end-time_start))

