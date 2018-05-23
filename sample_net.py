import os
import math
import json
import argparse
from bunch import Bunch
from copy import deepcopy
import torch
from torch.optim import lr_scheduler

from src.utils import load_model, load_data
from src.trainer import Trainer
from src.analysis import eval_accuracy

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('--dataset',default='fashionmnist')
argparser.add_argument('--save_dir', default='experiments/tmp/')
argparser.add_argument('--gpuid',default='0,')
argparser.add_argument('--momentum',type=float,default=0.0)
argparser.add_argument('--n_tries', type=int, default=1)
argparser.add_argument('--n_iters', type=int, default=[10000],nargs='+')
argparser.add_argument('--lr',type=float,default=[1e-1],nargs='+')
argparser.add_argument('--batch_size', type=int,default=1000)
argparser.add_argument('--arch', default='fnn')
argparser.add_argument('--loss',default='mse')
argparser.add_argument('--num_wrong_samples',type=int,default=0)
argparser.add_argument('--num_clean_samples',type=int,default=1000)
argparser.add_argument('--gain', type=float,default=1)
argparser.add_argument('--init', default='uniform')
argparser.add_argument('--save_model',default='')
argparser.add_argument('--tol',type=float,default=1e-3)
argparser.add_argument('--iter_display',type=int,default=200)
argparser.add_argument('--nclasses',type=int,default=10)
args = argparser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid


def weights_init(m,gain=1,init_type='uniform'):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        stddev = 1./math.sqrt(m.weight.size(1))
        if init_type == 'uniform':
            m.weight.data.uniform_(-gain*stddev,gain*stddev)
        else:
            m.weight.data.normal_(0,gain*stddev)

# Update Parameters
args_fashionmnist = {
    'num_clean_samples': 1000,
    'num_wrong_samples': 0,
    'load_size': 2000,
    'dataset': 'fashionmnist'
}

args_cifar10= {
    'num_clean_samples': 50000,
    'num_wrong_samples': 0,
    'arch': 'resnet',
    'load_size': 400
}

if args.dataset == 'fashionmnist':
    args_default = args_fashionmnist
else:
    args_default = args_cifar10
args = vars(args)
args_default.update(args)
print(json.dumps(args_default,indent=2))
args = Bunch(args_default)

if args.loss == 'mse':
    ct = torch.nn.MSELoss().cuda()
    one_hot = True
elif args.loss == 'hinge':
    ct = torch.nn.MultiMarginLoss(p=2)
    one_hot = False
elif args.loss == 'cross_entropy':
    print('Loss')
    ct = torch.nn.CrossEntropyLoss()
    one_hot = False


# Load data and model
for lr,n_iters in zip(args.lr,args.n_iters):
    for i in range(args.n_tries):
        print('==== Start of %d-th Experiment ==='%(i+1))

        trDL,teDL = load_data(args,one_hot=one_hot)
        net = load_model(args.dataset,args.arch)
        print(net)
        #net.apply(lambda t: weights_init(t,args.gain,args.init))

        optimizer = torch.optim.SGD(net.parameters(),lr = lr, momentum=args.momentum)
        # optimizer = torch.optim.RMSprop(net.parameters(),lr = lr)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                milestones = [10000000])
        trainer = Trainer(iter_display = args.iter_display)
        trainer.set_model(net,ct,optimizer,scheduler)
        res=trainer.train_sgd(
                            trDL,
                            batch_size = args.batch_size,
                            iter_start = 1,
                            iter_end = n_iters,
                            tol = args.tol
                        )

        trDL.reset()
        trL,trA,trC = eval_accuracy(net,ct,trDL)
        teL,teA,teC = eval_accuracy(net,ct,teDL)
        print('===> solution: ')
        print('\t train loss: %.2e, acc: %.2f'%(trL,trA))
        print('\t test loss: %.2e, acc: %.2f'%(teL,teA))

        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        filename = os.path.join(args.save_dir,
                    '%s_teA%.2f_lr%.2e_bz%d_momentum%.1f_try%d.pkl'%(
                                    args.dataset,teA, lr, args.batch_size,args.momentum,i+1)
                )
        torch.save(net.state_dict(),filename)

        if args.save_model != '':
            torch.save(net.state_dict(), args.save_model)
        print('==== End of %d-th Experiment ==='%(i+1))
