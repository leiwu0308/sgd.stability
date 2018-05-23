from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .trainer import accuracy
from .net_utils import flatten_gradients, num_parameters
from .linalg import Hv_batch


def stability_index_sgd_v(net,ct,dataloader,v):
    X,y = dataloader.X,dataloader.y
    n_samples = len(y)
    V, M, Hv = 0,0,0
    for i in range(n_samples):
        bx = Variable(X[i:i+1].cuda())
        by = Variable(y[i:i+1].cuda())

        Hi_v = Hv_batch(net,ct,bx,by,v)
        Hv += Hi_v
        V += torch.dot(Hi_v,Hi_v)
    Hv /= n_samples
    M = torch.dot(v,Hv)
    V = V/n_samples - M*M
    return M,sqrt(V)


def eval_accuracy(model,ct,dataloader):
    model.eval()
    loss,acc,confidence = 0.0,0.0,0.0
    nbatch = 0
    for imgs,labels in dataloader:
        X = Variable(imgs.cuda(),volatile=True)
        y = Variable(labels.cuda(),volatile=True)

        logit = model(X)
        loss += ct(logit,y).data[0]
        P = F.softmax(logit,dim=1).data
        p,pred_y = torch.max(P,1)
        acc += accuracy(logit.data,y.data)
        confidence += p.mean()
        nbatch += 1

    return loss/nbatch, acc/nbatch, 100*confidence/nbatch

def compute_loss_distribution(net,ct,dataloader):
    net.zero_grad()
    res = []
    reduce_state = ct.reduce
    ct.reduce = False
    for imgs,labels in dataloader:
        X = Variable(imgs.cuda())
        y = Variable(labels.cuda())

        logit = model(X)
        E = ct(logit,y)
        res = res + list(E.data)
    ct.reduce = reduce_state
    return res


def compute_gradients(net,ct,X,y):
    """
    return n_samples x num_parameters FloatTensor
    """
    n_parameters = num_parameters(net)
    gradients = []

    for i in range(n_samples):
        bx = Variable(X[i:i+1].cuda())
        by = Variable(y[i:i+1].cuda())

        net.zero_grad()
        logit = net(bx)
        E = ct(logit,by)
        E.backward()

        # concat gradient
        gradients.append(flatten_gradient(net))

    return gradients


def compute_gradient_diversity(net,ct,X,y):
    gradient_total = torch.zeros(
                        num_parameters(net)
                    )
    gradient_norm_total = 0
    n_samples = len(y)

    for i in range(n_samples):
        bx = Variable(X[i:i+1].cuda())
        by = Variable(y[i:i+1].cuda())

        net.zero_grad()
        logit = net(bx)
        E = ct(logit,by)
        E.backward()

        gradient = flatten_gradients(net)
        gradient_total += gradient
        gradient_norm_total += gradient.norm()**2

    grad_mean = gradient_total / n_samples
    grad_var = gradient_norm_total / n_samples
    return grad_var/grad_mean.norm()**2


def weight_norm(net,p=2):
    res = 0
    for para in net.parameters():
        res += para.data.abs().pow(p).sum()
    res = res**(1.0/p)
    return res

