import os
import math
import json
import argparse
from bunch import Bunch
from copy import deepcopy
import torch
from torch.optim import lr_scheduler

from src.utils import load_model, load_data, parse_args
from src.trainer import Trainer
from src.analysis import eval_accuracy

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

def weights_init(m,gain=1,init_type='uniform'):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        stddev = 1./math.sqrt(m.weight.size(1))
        if init_type == 'uniform':
            m.weight.data.uniform_(-gain*stddev,gain*stddev)
        else:
            m.weight.data.normal_(0,gain*stddev)


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
        net = load_model(args.dataset,args.arch,width=args.width,depth=args.depth)
        print(net)
        #net.apply(lambda t: weights_init(t,args.gain,args.init))

        optimizer = torch.optim.SGD(net.parameters(),lr = lr, momentum = args.momentum)
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                milestones = args.decay)
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


        # Save model
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        filename = os.path.join(args.save_dir,
                    '%s_teA%.2f_lr%.2e_bz%d_momentum%.1f_try%d.pkl'%(
                                    args.dataset,teA, lr, args.batch_size,args.momentum,i+1)
                )
        torch.save(net.state_dict(),filename)
        print('==== End of %d-th Experiment ==='%(i+1))
