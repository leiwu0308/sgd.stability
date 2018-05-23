import os
import time
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Trainer:
    def __init__(self,iter_display=100,iter_ckpt=10,store_ckpt=False,dir_ckpt='checkpoints'):
        self.iter_display = iter_display
        self.iter_ckpt = iter_ckpt
        self.store_ckpt = store_ckpt
        self.dir_ckpt = dir_ckpt
        self.history = []


    def set_model(self,model,ct,optimizer,scheduler=None):
        self.model = model
        self.ct = ct 
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_sgd(self,dataloader,batch_size=-1,iter_start=0,iter_end=1000,tol=-1):
        self.model.train()
        if batch_size == -1:
            batch_size = dataloader.nsamples

        since = time.time()
        global _ACC_AVG
        global _LOSS_AVG
        _ACC_AVG = -1
        _LOSS_AVG = -1
        for iter_now in range(iter_start,iter_end):
            self.scheduler.step()
            current_lr = self.scheduler.get_lr()[0]
            
            def closure():
                global _ACC_AVG
                global _LOSS_AVG
                self.optimizer.zero_grad()
                loss,acc = comp_gradient(self.model,self.ct,dataloader,batch_size)
                _ACC_AVG = 0.9 * _ACC_AVG + 0.1 * acc if _ACC_AVG > 0 else acc
                _LOSS_AVG = 0.9 * _LOSS_AVG + 0.1 * loss.data[0] if _LOSS_AVG > 0 else loss.data[0]
                return loss 

            # update information
            if iter_now%self.iter_ckpt == 0 and self.store_ckpt:
                if not os.path.exists(self.dir_ckpt):
                    pathlib.Path(self.dir_ckpt).mkdir(parents=True,exist_ok=True)
                torch.save(self.model.state_dict(),
                        '%s/batchsize_%d_start_%d_now_%d_end_%d.ckpt'%(
                                self.dir_ckpt,batch_size,iter_start,iter_now,iter_end
                            )
                    )
            if iter_now%self.iter_display == 0:
                now = time.time()
                print('%d/%d, lr=%.2e, took %.0f seconds, trL: %.2e, trA: %.2f'%(
                        iter_now,iter_end,current_lr,now-since,_LOSS_AVG,_ACC_AVG))
                since = time.time()

            # perform (stochastic) gradient descent
            self.optimizer.step(closure)

            if _LOSS_AVG < tol:
                print('achive required error with loss: %.2e'%(_LOSS_AVG))
                break
            
def comp_gradient(model,ct,dataloader,batch_size):
    loss,acc = 0,0
    nbatch = batch_size // dataloader.batch_size

    model.zero_grad()
    for i in range(nbatch):
        imgs,labels = next(dataloader)
        X = Variable(imgs.cuda())
        y = Variable(labels.cuda())

        logit = model(X)
        E = ct(logit,y)
        P = F.softmax(logit,dim=1).data
        E.backward()
        
        loss += E
        acc += accuracy(logit.data,y.data)

    for p in model.parameters():
        p.grad.data /= nbatch

    return loss/nbatch,acc/nbatch


def accuracy(logit, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = logit.shape[0]
    if target.ndimension() == 2:
        _, y_true = torch.max(target,1)
    else:
        y_true = target 
    _, y_pred = torch.max(logit,1)
    acc = (y_true==y_pred).sum()*100.0/batch_size 
    return acc













