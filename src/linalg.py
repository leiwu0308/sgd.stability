import time
import torch
import torch.autograd as autograd

from .net_utils import flatten_parameters, \
                set_parameters, flatten_gradients, \
                num_parameters


def eigen_variance(net,ct,dataloader,niters=10,tol=1e-2,verbose=False):
    n_parameters = num_parameters(net)
    v0 = torch.randn(n_parameters)
    v0 /= v0.norm()

    Av_func = lambda v: variance_vec_prod(net, ct, dataloader, v)
    mu = power_method(v0, Av_func, niters, tol, verbose)
    return mu

def eigen_hessian(net,ct,dataloader,niters=10,tol=1e-2,verbose=False):
    n_parameters = num_parameters(net)
    v0 = torch.randn(n_parameters)
    v0 /= v0.norm()

    Av_func = lambda v: hessian_vec_prod(net,ct,dataloader,v)
    mu = power_method(v0, Av_func, niters, tol, verbose)
    return mu

def variance_vec_prod(net,ct,dataloader,v):
    X, y = dataloader.X, dataloader.y
    Av, Hv, nS = 0, 0, len(y)
    for i in range(nS):
        bx = autograd.Variable(X[i:i+1].cuda())
        by = autograd.Variable(y[i:i+1].cuda())
        Hi_v = Hv_batch(net, ct, bx, by, v)
        Hi2_v = Hv_batch(net, ct, bx, by, Hi_v)
        Av += Hi2_v
        Hv += Hi_v
    Av /= nS
    Hv /= nS
    H2v = hessian_vec_prod(net, ct, dataloader, Hv, gpu=True)
    Av = Av - H2v
    return Av

def hessian_vec_prod(net,ct,dataloader,v, gpu=True):
    Hv = 0
    n_batch = 0
    for bx,by in dataloader:
        bx = bx.cuda()
        by = by.cuda()
        bx = autograd.Variable(bx)
        by = autograd.Variable(by)
        t = Hv_batch(net,ct,bx,by,v)
        Hv += t
        n_batch += 1
    Hv /= n_batch
    return Hv

## helper functions
def Hv_batch(net,ct,batch_x,batch_y,v):
    net.eval()
    logits = net(batch_x)
    loss = ct(logits,batch_y)

    grads = autograd.grad(loss,net.parameters(),create_graph=True,retain_graph=True)
    idx = 0
    res = 0
    for g in grads:
        ng = torch.numel(g)
        vg = v[idx:idx+ng].view(g.shape)
        vg = autograd.Variable(vg.cuda())
        idx += ng
        res += torch.dot(vg,g)

    Hv = autograd.grad(res,net.parameters())
    Hv = [t.data.cpu().view(-1) for t in Hv]
    Hv = torch.cat(Hv)
    return Hv

def power_method(v0,Av_func,niters=10,tol=1e-3,verbose=False):
    v = v0
    mu = 0
    for i in range(niters):
        time_start = time.time()
        mu_pre = mu
        Av = Av_func(v)
        mu = torch.dot(Av,v)/torch.dot(v,v)
        v = Av/Av.norm()

        if abs(mu-mu_pre)/abs(mu) < tol:
            break
        if verbose:
            print('%d-th step takes %.0f seconds, \t %.2e'%(i+1,time.time()-time_start,mu))
    return mu

def ssd_method(v0,Hv_func,dt,niters=10):
    v = v0
    for i in range(niters):
        Hv = Hv_func(v)
        mu = torch.dot(v,Hv)
        v = v + 2*dt*(Hv-mu*v)
        v /= v.norm()
        print(mu)
    return mu

