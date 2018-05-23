import argparse
import os
import re
import json
from bunch import Bunch

import torch
from torch.autograd import Variable

from .models.cifar import resnet
from .models.cifar import lenet as lenet_cifar
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

def load_model(dataset,arch=None):
    if dataset == 'fashionmnist':
        if arch is None or arch == 'fnn':
            return resfnn(depth=3,width=500).cuda()
        elif arch == 'lenet':
            return lenet().cuda()
        else:
            raise ValueError('The network architecture is not supported')

    elif dataset == 'cifar10':
        if arch is None or arch == 'resnet':
            return resnet(width=4,depth=8,num_classes=2).cuda()
        elif arch == 'vgg':
            return vgg11(num_classes=2).cuda()
        elif arch == 'lenet':
            return lenet_cifar(num_classes=2).cuda()
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
