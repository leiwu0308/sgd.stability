import os
import argparse
import time
import pickle
import json
from math import sqrt
from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib as mpl

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
import matplotlib.pyplot as plt

from bunch import Bunch
# import seaborn as sns
# sns.set()

import torch
import torch.nn as nn

def list2dict(data):
    data = sorted(data,key=lambda t:t[0])
    res = defaultdict(list)
    for v in data:
        lr = v[0]
        res[lr].append(v[1:])
    res = OrderedDict(res)
    return res

def parse_data(data,batch_size=None):
    """
    Convert list to  dict
    """
    data = list2dict(data)
    res = OrderedDict()
    for key,values in data.items():
        values= list2dict(values)

        R = {'batch_size':[], 'test_accs': [],
                     'sharpness': [], 'diversity': []}
        for k,v in values.items():
            if (len(batch_size) !=0) and (not k in batch_size):
                continue

            v = np.asarray(v)
            mean = v.mean(axis=0)
            std = v.std(axis=0)

            R['batch_size'].append(k)
            R['sharpness'].append([mean[0],std[0]])
            R['diversity'].append([mean[1],std[1]])
            R['test_accs'].append([mean[3],std[3]])

        res[key] = R
    return res

def plot(data,title,save_dir=None):
    plt.figure()
    for lr, values in data.items():
        x_R = values['batch_size']
        y_R = np.asarray(values['sharpness'])
        plt.errorbar(np.log10(x_R), y_R[:,0], yerr = y_R[:,1],
            marker='o',markersize=10, lw=6, label=r'$\eta$=%.2f'%(lr))
    plt.legend()
    plt.xlabel(r'$\log_{10}$(batch size)',fontsize=35)
    plt.ylabel(r'sharpness',fontsize=35)
    plt.title(title,fontsize=30)
    plt.savefig('results/%s_bz_vs_sharpness.pdf'%(title),bbox_inches='tight')

    plt.figure()
    for lr, values in data.items():
        x_R = values['batch_size']
        y_R = np.asarray(values['test_accs'])
        print(y_R[:,0],y_R[:,1])
        plt.errorbar(np.log10(x_R), y_R[:,0], yerr = y_R[:,1],
            marker='o',markersize=10, lw=6, label='lr=%.2f'%(lr))
    # plt.ylim([78,81.5])
    plt.legend()
    plt.xlabel(r'$\log_{10}$(batch size)',fontsize=35)
    plt.ylabel(r'test accuracy(\%)',fontsize=35)
    plt.title(title,fontsize=30)
    plt.savefig('results/%s_bz_vs_testacc.pdf'%(title),bbox_inches='tight')

    plt.figure()
    for lr, values in data.items():
        x_R = values['batch_size']
        y_R = np.asarray(values['diversity'])
        plt.errorbar(np.log10(x_R), y_R[:,0], yerr = y_R[:,1],
            marker='o',markersize=10, lw=6, label='lr=%.2f'%(lr))
    plt.legend()
    plt.xlabel(r'$\log_{10}$(batch size)',fontsize=35)
    plt.ylabel(r'nonuniformity',fontsize=35)
    plt.title(title,fontsize=30)
    plt.savefig('results/%s_bz_vs_nonuniformity.pdf'%(title),bbox_inches='tight')



def scatter_plot(xR,yR,title=None,save=None):
    plt.figure()
    plt.plot(xR,yR,'ok',markersize=8,label='solutions')
    plt.xlabel(r'sharpness',fontsize=35)
    plt.ylabel(r'nonuniformity',fontsize=35)
    if title is not None:
        plt.title(title,fontsize=30)

    if save is not None:
        plt.savefig(save,bbox_inches='tight')

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--data', default='')
    argparser.add_argument('--scatter',action='store_true')
    argparser.add_argument('--gd',action='store_true')
    argparser.add_argument('--title',default=None)
    argparser.add_argument('--save',default=None)
    argparser.add_argument('--lr',type=float,default=[1],nargs='+')
    argparser.add_argument('--batch_size', type=float, default=[], nargs='+')
    args = argparser.parse_args()

    with open(args.data,'rb') as f:
        data = pickle.load(f)

    if args.gd:
        data = parse_data(data)
        print(json.dumps(data,indent=2))
        return
    if args.scatter:
        data = np.asarray(data)
        scatter_plot(data[:,2], data[:,3],args.title,args.save)
    else:
        data = parse_data(data,args.batch_size)
        res = {}
        for v in args.lr:
            res[v] = data[v]
        plot(res,args.title,args.save)



if __name__ == '__main__':
    main()
