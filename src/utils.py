import argparse
import re
import json
from bunch import Bunch

import torch
from torch.autograd import Variable

from .models.cifar import resnet
from .models.vgg import vgg11, vgg11_big
from .models.mnist import resfnn, lenet
from .data import load_cifar,load_mnist


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='configs/base.json',
        help='The Configuration file')
    args = argparser.parse_args()
    config = process_config(args.config)
    if config.load_size > config.batch_size:
        config.load_size = config.batch_size

    return config

def load_data(args,stop=False,one_hot=False,root=None):
    if args.load_size > args.batch_size:
        args.load_size = args.batch_size
    if args.dataset[0:5] == 'cifar':
        trDL, teDL = load_cifar(
                            root = root,
                            batch_size = args.load_size,
                            nclasses=args.nclasses,
                            nsamples = args.num_clean_samples,
                            num_wrong = args.num_wrong_samples,
                            stop = stop,
                            one_hot = one_hot
                        )
    elif args.dataset == 'fashionmnist':
        trDL,teDL = load_mnist(
                            root = root,
                            fashion=True,
                            batch_size = args.load_size,
                            nsamples = args.num_clean_samples,
                            num_wrong = args.num_wrong_samples,
                            stop = stop,
                            one_hot = one_hot
                        )
    else:
        raise ValueError('The specified dataset has not been implemented')
    return trDL,teDL

def load_model(dataset,arch=None,width=500,depth=2):
    if dataset == 'fashionmnist':
        if arch is None or arch == 'fnn':
            return resfnn(depth=depth,width=width).cuda()
        elif arch == 'lenet':
            return lenet().cuda()
        else:
            raise ValueError('The network architecture is not supported')

    elif dataset == 'cifar10':
        if arch is None or arch == 'resnet':
            return resnet(width=4,depth=8,num_classes=2).cuda()
        elif arch == 'vgg':
            return vgg11(num_classes=2).cuda()
        elif arch == 'bigvgg':
            return vgg11_big().cuda()
        else:
            raise ValueError('The network architecture is not supported')
    else:
        raise ValueError('The dataset %s is not supported'%(dataset))

def get_train_info(filename):
    filename = os.path.basename(filename)
    fp_info = re.split('\_|\.',filename)
    batch_size = int(fp_info[1])
    iter_start = int(fp_info[3])
    iter_now = int(fp_info[5])
    iter_end = int(fp_info[7])

    return iter_start,iter_now,iter_end,batch_size

def get_config_from_json(json_file):
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)

    return config


def process_config(json_file):
    config = get_config_from_json(json_file)
    config.summary_dir = os.path.join("experiments",
                                      config.exp_name,
                                      "summary/")
    config.ckpt_dir = os.path.join("experiments",
                                   config.exp_name,
                                   "checkpoint/")
    config.data_dir = os.path.join("experiments",
                                   config.exp_name)
    return config

def parse_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--save_dir', default='experiments/tmp/')
    argparser.add_argument('--save_model', default='')
    argparser.add_argument('--gpuid', default='0,')
    argparser.add_argument('--iter_display',type=int,default=200)
    argparser.add_argument('--n_tries', type=int, default=1)

    argparser.add_argument('--dataset', default='fashionmnist')
    argparser.add_argument('--nclasses', type=int, default=10)
    argparser.add_argument('--num_wrong_samples', type=int, default=0)
    argparser.add_argument('--num_clean_samples', type=int, default=1000)
    argparser.add_argument('--batch_size', type=int, default=1000)

    argparser.add_argument('--arch', default='fnn')
    argparser.add_argument('--width', type=int, default=500)
    argparser.add_argument('--depth', type=int, default=2)
    argparser.add_argument('--loss', default='mse')

    argparser.add_argument('--momentum',type=float,default=0.0)
    argparser.add_argument('--lr',type=float,default=[1e-1],nargs='+')
    argparser.add_argument('--n_iters', type=int, default=[10000],nargs='+')
    argparser.add_argument('--decay',type=int,default=[5000000],nargs='+')
    argparser.add_argument('--init', default='uniform')
    argparser.add_argument('--gain', type=float,default=1)
    argparser.add_argument('--tol',type=float,default=1e-3)

    argparser.add_argument('--nonuniformity', action='store_true',
                help='If True, compute the non-uniformity')
    argparser.add_argument('--model', default='', help='pickle files of the model')
    argparser.add_argument('--save_res',default='',help='file name to store results')
    args = argparser.parse_args()
    if args.batch_size == -1:
        args.batch_size = args.num_clean_samples + args.num_wrong_samples


    # Update Parameters
    args_fashionmnist = {
        'num_clean_samples': 1000,
        'num_wrong_samples': 0,
        'arch': 'fnn',
        'load_size': 2000,
    }

    args_cifar10= {
        'num_clean_samples': 1000,
        'num_wrong_samples': 0,
        'arch': 'resnet',
        'load_size': 100
    }

    if args.dataset == 'fashionmnist':
        args_default = args_fashionmnist
    else:
        args_default = args_cifar10
    args = vars(args)
    args_default.update(args)

    print('===> CONFIGS:')
    print(json.dumps(args_default,indent=2))
    args = Bunch(args_default)
    return args
