import os
import argparse
import torch
from torch.optim import lr_scheduler

from src.trainer import Trainer
from src.analysis import eval_accuracy
from src.utils import get_args,load_model,load_data,get_train_info

argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument('--gpuid',default='0,')
argparser.add_argument('--arch', default='fnn')
argparser.add_argument('--restart_file', default=None)
argparser.add_argument('--loss',default='mse')
argparser.add_argument('--dataset',default='fashionmnist')
argparser.add_argument('--load_size', type=int,default=1000)
argparser.add_argument('--num_wrong_samples',type=int,default=0)
argparser.add_argument('--num_clean_samples',type=int,default=1000)
argparser.add_argument('--nclasses', type=int, default=2)

argparser.add_argument('--n_iters', type=int, default=10000)
argparser.add_argument('--batch_size', type=int,default=1000)
argparser.add_argument('--momentum',type=float,default=0.0)
argparser.add_argument('--lr',type=float,default=1e-1)

argparser.add_argument('--dir_ckpt', default='experiments/checkpoints')
argparser.add_argument('--iter_ckpt', type=int,default=10)
argparser.add_argument('--store_ckpt', action='store_true')
argparser.add_argument('--store_last_ckpt', action='store_true')
argparser.add_argument('--iter_display', type=int,default=100)

args = argparser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid

print(args)
#== [2] load data and define model
if args.loss == 'mse':
    ct = torch.nn.MSELoss().cuda()
    one_hot = True
elif args.loss == 'hinge':
    ct = torch.nn.MultiMarginLoss(p=2)
    one_hot = False
elif args.loss == 'cross_entropy':
    ct = torch.nn.CrossEntropyLoss()
    one_hot = False

print('===> Start Training')
trDL,teDL = load_data(args,one_hot=one_hot)
net = load_model(args.dataset,args.arch)
optimizer = torch.optim.SGD(net.parameters(),
                        lr = args.lr,
                        momentum = args.momentum
                    )
scheduler = lr_scheduler.MultiStepLR(optimizer,
                        milestones = [10000000000000],
                        gamma = 0.1
                    )

if args.restart_file is not None:
    net.load_state_dict(torch.load(args.restart_file))

#== [3] Training network
trainer = Trainer(
                iter_display = args.iter_display,
                iter_ckpt = args.iter_ckpt,
                store_ckpt = args.store_ckpt,
                dir_ckpt = args.dir_ckpt,
            )
trainer.set_model(net,ct,optimizer,scheduler)
res=trainer.train_sgd(
                    trDL,
                    batch_size = args.batch_size,
                    iter_start = 0,
                    iter_end = args.n_iters
                )

#  [4] Store Model
trDL.reset(trDL.y.shape[0])
trL,trA,trC = eval_accuracy(net,ct,trDL)
teL,teA,teC = eval_accuracy(net,ct,teDL)
print('===> solution: ')
print('\t train loss: %.2e, acc: %.2f'%(trL,trA))
print('\t test loss: %.2e, acc: %.2f'%(teL,teA))

if args.store_last_ckpt:
    filename = 'experiments/tmp/%s_teA%.2f_lr%.2e_bz%d.pkl'%(
                        args.dataset,teA,args.lr,args.batch_size)
    torch.save(net.state_dict(),filename)

