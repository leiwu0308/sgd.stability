import os
import time
import pickle 
import argparse
import torch 
import torch.nn as nn

from src.utils import load_model,load_data
from src.analysis import eval_accuracy
from src.folderwalker import FolderWalker
from src.linalg import eigen_hessian, eigen_variance

def compute_norm(net,args):
    walker = FolderWalker(net)
    norms = walker.comp_norm(args.dir)
    with open(os.path.join(args.save_dir,'norm.pkl'),'wb') as f:
        pickle.dump(norms, f)

def compute_loss(net,ct,trDL,teDL,args):
    walker = FolderWalker(net, ct)
    train_loss = walker.run(
                    args.dir,
                    trDL,
                    eval_accuracy,
                )
    test_loss = walker.run(
                    args.dir,
                    teDL,
                    eval_accuracy
                )
    loss = {
        'train': train_loss['traj'],
        'test': test_loss['traj'],
        'batch_size': train_loss['batch_size']
    }
    with open(os.path.join(args.save_dir,'loss.pkl'),'wb') as f:
        pickle.dump(loss,f)

def compute_diversity(net,ct,trDL,args):
    walker = FolderWalker(net, ct)
    diversity = walker.run(
                    args.dir,
                    trDL,
                    compute_gradient_diversity
                )

    with open(os.path.join(args.save_dir,'diversity.pkl'),'wb') as f:
        pickle.dump(diversity,f)

def compute_sharpness(net,ct,trDL,args):
    func = lambda net,ct,dataloader: eigen_hessian(net,ct,dataloader,niters=50,tol=1e-4)
    walker = FolderWalker(net,ct)
    sharpness = walker.run(
                    args.dir, 
                    trDL,
                    func,
                    verbose=True
                )
    with open(os.path.join(args.save_dir,'sharpness.pkl'),'wb') as f:
        pickle.dump(sharpness,f)

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--gpuid',default='0,')
    argparser.add_argument('--arch', default='fnn')
    argparser.add_argument('--loss',default='mse')
    argparser.add_argument('--dataset',default='fashionmnist')
    argparser.add_argument('--load_size', type=int,default=1000)
    argparser.add_argument('--num_wrong_samples',type=int,default=0)
    argparser.add_argument('--num_clean_samples',type=int,default=1000)
    argparser.add_argument('--batch_size', type=int,default=1000)
    argparser.add_argument('--nclasses', type=int,default=2)

    argparser.add_argument('--task', default='loss')
    argparser.add_argument('--dir', default='')
    argparser.add_argument('--save_dir', default='')

    args = argparser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuid
    print(args)

    if args.loss == 'mse':
        ct = torch.nn.MSELoss().cuda()
        one_hot = True
    elif args.loss == 'hinge':
        ct = torch.nn.MultiMarginLoss(p=2)
        one_hot = False
    elif args.loss == 'cross_entropy':
        ct = torch.nn.CrossEntropyLoss()
        one_hot = False
    
    trDL,teDL = load_data(args, stop=True, one_hot=one_hot)
    net = load_model(args.dataset, args.arch)


    if args.task == 'loss':
        compute_loss(net, ct, trDL, teDL,args)
    elif args.task == 'nonuniformity':
        compute_diversity(net,ct,trDL,args)
    elif args.task == 'weight_norm':
        compute_norm(net,args)
    elif args.task == 'sharpness':
        compute_sharpness(net, ct, trDL, args)
    else:
        raise ValueError('Task %s has not been implemented'%(args.task))

if __name__ == '__main__':
    main()






