import os
import time
import torch
import argparse
from copy import deepcopy
from bunch import Bunch
# torch.manual_seed(20)

from src.utils import load_model, load_data, parse_args
from src.net_utils import num_parameters
from src.trainer import Trainer
from src.analysis import eval_accuracy, weight_norm
from src.linalg import *

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

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
net = load_model(args.dataset,args.arch, width=args.width, depth=args.depth)
net.load_state_dict(torch.load(args.model))

# Evaluation
trL,trA,trC = eval_accuracy(net,ct,trDL)
teL,teA,teC = eval_accuracy(net,ct,teDL)
print('===> SOLUTION INFO: ')
print('\t train loss: %.2e, acc: %.2f'%(trL,trA))
print('\t test loss: %.2e, acc: %.2f'%(teL,teA))
print('l2 norm %.2e\n'%(weight_norm(net)))

print('===> COMPUTE SHARPNESS:')
time_start = time.time()
mu = eigen_hessian(net,ct,trDL,verbose=True,tol=1e-4,niters=10)
time_end = time.time()
print('takes %.0f seconds\n'%(time_end-time_start))

if args.nonuniformity:
    print('===> COMPUTE NON-UNIFORMITY:')
    time_start = time.time()
    mu = eigen_variance(net,ct,trDL,verbose=True,tol=1e-4,niters=10)
    time_end = time.time()
    print('takes %.0f seconds'%(time_end-time_start))

