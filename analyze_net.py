import os
import pickle
import argparse
from collections import defaultdict
from bunch import Bunch

import torch
from src.utils import load_model, load_data
from src.folderwalker import scan
from src.net_utils import num_parameters


args_mnist = {
    'num_clean_samples': 1000,
    'num_wrong_samples': 0,
    'arch': 'fnn',
    'load_size': 1000,
    'dataset': 'fashionmnist',
    'batch_size': 1000,
}

args_cifar = {
    'num_clean_samples': 1000,
    'num_wrong_samples': 0,
    'arch': 'resnet',
    'load_size': 1000,
    'dataset': 'cifar10',
    'batch_size': 1000
}

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',default='0,')
    argparser.add_argument('--dataset',default='fashionmnist')
    argparser.add_argument('--arch',default='fnn')
    argparser.add_argument('--dir', default='')
    argparser.add_argument('--save',default='results/tmp.pkl')
    argparser.add_argument('--nonuniformity',action='store_true')
    argparser.add_argument('--num_wrong_samples',type=int,default=0)
    argparser.add_argument('--num_clean_samples',type=int,default=1000)
    argparser.add_argument('--loss',default='mse')
    argparser.add_argument('--nclasses', type=int,default=2)
    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

    if args.dataset == 'fashionmnist':
        args_default = args_mnist
    else:
        args_default = args_cifar
    args = vars(args)
    args_default.update(args)
    args = Bunch(args_default)

    return args

def main():
    args = parse_args()
    trDL,teDL = load_data(args,stop=True,one_hot=True)
    net = load_model(args.dataset,args.arch)
    ct = torch.nn.MSELoss()
    print(net)
    print(num_parameters(net))

    res = scan(net,ct,trDL,teDL,args.dir,niters=50,verbose=True,variance_comp=args.nonuniformity,niters_var=25)
    with open(args.save,'wb') as f:
        pickle.dump(res,f)

if __name__ == '__main__':
    main()
