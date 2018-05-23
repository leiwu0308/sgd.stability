import torch
import torch.nn as nn


def flatten_gradients(net):
    parameters = [ t.grad.data.cpu().view(-1) for t in net.parameters()]
    return torch.cat(parameters)

def flatten_parameters(net):
    parameters = [ t.data.cpu().view(-1) for t in net.parameters()]
    return torch.cat(parameters)

def num_parameters(net):
    n_parameters = 0
    for para in net.parameters():
        n_parameters += para.data.numel()

    return n_parameters

def set_parameters(net,paras):
    # copy_ automatically handles cpu -> gpu
    idx = 0
    for p in net.parameters():
        n_p = p.data.view(-1).shape[0]
        shape = p.data.shape
        p.data.copy_(paras[idx:idx+n_p].view(shape))
        idx += n_p