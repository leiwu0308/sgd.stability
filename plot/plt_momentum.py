import os
import pickle
import argparse
import json
import math
from collections import defaultdict, OrderedDict
import numpy as np
import matplotlib as mpl 

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=25
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.titlesize'] = 20
import matplotlib.pyplot as plt


def list2dict(keys,values):
    res = defaultdict(list)
    for k,v in zip(keys,values):
        res[k].append(v)
    res = OrderedDict(res)
    return res

def read_data(filename,batch_size,lr=0.1):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    # sort data according to momentum
    data = np.asarray(sorted(data,key=lambda t:t[2]))
    # extract data according to batch size
    idx = (data[:,0]==lr) & (data[:,1]==batch_size) & (data[:,2] <= 0.92)
    data = data[idx,:]

    momen = data[:,2]
    sharpn = data[:,3]
    noniform = data[:,4]

    sharpn = list2dict(momen,sharpn)
    noniform = list2dict(momen,noniform)

    return sharpn, noniform

def plot(data,label='',marker='o',linestyle='-',color=''):
    x = list(data.keys())
    y_mean,y_std = [],[]
    for v in data.values():
        # v = np.log10(v)
        y_mean.append(np.mean(v))
        y_std.append(np.std(v))

    plt.errorbar(x, y_mean, yerr = y_std, linestyle=linestyle,marker=marker,
            barsabove=True,capsize=10, markersize=8, capthick=2,lw=5,label=label)

#------------------------------------
x_ticks = [0,0.2,0.4,0.6,0.8,0.9]

sharpn_fmnist_SGD,noniform_fmnist_SGD = read_data('data/fmnist_momentum.pkl', 10, 0.1)
sharpn_fmnist_GD,noniform_fmnist_GD = read_data('data/fmnist_momentum.pkl', 1000, 0.1)                         
fig,ax = plt.subplots(1,1)
plot(noniform_fmnist_SGD,label='SGD')
plot(noniform_fmnist_GD,label='GD',marker='o',linestyle='--')
plt.xlabel('Momentum')
plt.ylabel('Non-uniformity')
plt.title('FashionMNIST')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)
plt.legend()
plt.savefig('figures/fmnist_momentum_nonuniformity.pdf',bbox_inches='tight')



fig,ax = plt.subplots(1,1)
plot(sharpn_fmnist_SGD,label='SGD')
plot(sharpn_fmnist_GD,label='GD',marker='o',linestyle='--')
plt.xlabel('Momentum')
plt.ylabel('Sharpness')
plt.title('FashionMNIST')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)
plt.legend()
plt.savefig('figures/fmnist_momentum_sharpness.pdf',bbox_inches='tight')



#------------------------------------
sharpn_cifar_SGD,noniform_cifar_SGD = read_data('data/cifar10_momentum.pkl', 10, 0.01)
sharpn_cifar_GD, noniform_cifar_GD = read_data('data/cifar10_momentum.pkl', 1000, 0.01)

fig,ax = plt.subplots(1,1)
plot(noniform_cifar_SGD,label='SGD')
plot(noniform_cifar_GD,label='GD',marker='o',linestyle='--')
plt.xlabel('Momentum')
plt.ylabel('Non-uniformity')
plt.title('CIFAR-10')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)
plt.legend()
plt.savefig('figures/cifar_momentum_nonuniformity.pdf',bbox_inches='tight')

fig,ax = plt.subplots(1,1)
plot(sharpn_cifar_SGD,label='SGD')
plot(sharpn_cifar_GD,label='GD',marker='o',linestyle='--')
plt.xlabel('Momentum')
plt.ylabel('Sharpness')
plt.title('CIFAR-10')
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks)
plt.legend()
plt.savefig('figures/cifar_momentum_sharpness.pdf',bbox_inches='tight')