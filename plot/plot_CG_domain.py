import os
import pickle
import argparse
import json
from collections import defaultdict
import numpy as np
import matplotlib as mpl 

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=17
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()


def plot_cifar():
    data = 'results/data/cifar10/cifar_sgd_range.pkl'
    with open(data,'rb') as f:
        data = pickle.load(f)
    res = defaultdict(list)
    print('learning rate is',data[0][0])
    for v in data:
        bz = v[1]
        res[bz].append(v[2:4])
    print(json.dumps(res,indent=2))

    plot(res,0.1,'results/figures/cifar10_sgd_range.pdf','CIFAR10')
#=====================================================

def plot_fmnist():
    data = 'results/data/fmnist/lr0.5_range_n1000_w0.pkl'
    with open(data,'rb') as f:
        data = pickle.load(f)
    res = defaultdict(list)
    for v in data:
        bz = v[1]
        res[bz].append(v[2:4])
    print(json.dumps(res,indent=2))

    plot(res, 0.5,'results/figures/fmnist_sgd_range.pdf','FashionMNIST')


def plot(res,lr,save_fig=None,title=None):
    for k in res.keys():
        res[k] = np.asarray(res[k])
    v1000,v25,v10,v4 = res[1000],res[25],res[10],res[4]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    plt.figure()
    plt.plot(v1000[:,0],v1000[:,1],'o',label='GD',markersize=12,color=colors[0])
    plt.plot(v25[:,0],v25[:,1],'*',label='SGD, B=25',markersize=12,color=colors[1])
    plt.plot(v10[:,0],v10[:,1],'s',label='SGD, B=10',markersize=12,color=colors[2])
    plt.plot(v4[:,0],v4[:,1],'^',label='SGD, B=4',markersize=12,color=colors[3])
    plt.legend()

    # GD range
    y_R = np.linspace(0, v1000.max()+5,100)
    x_R = np.ones(100)*2/lr
    plt.plot(x_R,y_R,'-k')
    plt.xlim([0,2/lr+0.2])
    plt.ylim([0,v1000.max()+2])




    n = 1000
    for i,bz in enumerate([25,10,4]):
        x_R = np.linspace(0,2/lr,100)
        # func = lambda t: 2*
        y_R = np.ones(100)*np.sqrt(1.0*bz*(n-1)/(n-bz))/lr
        plt.plot(x_R,y_R,'--k',color=colors[i+1],lw=4)

    plt.xlabel(r'sharpness',fontsize=35)
    plt.ylabel(r'nonuniformity',fontsize=35)
    if title is not None:
        plt.title(title,fontsize=30)

    plt.text(1.8/lr, 1/lr, r'$2/\eta$',fontsize=20)
    plt.savefig(save_fig, bbox_inches='tight')


if __name__ == '__main__':
    plot_fmnist()


