import os
import pickle
import argparse
import json
import math
from collections import defaultdict
import numpy as np
import matplotlib as mpl 

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['axes.labelsize'] = 20
# mpl.rcParams['axes.labelsize'] = 20
import matplotlib.pyplot as plt


def read_data(filename,batch_size,lr=0.1):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    # sort data according to momentum
    data = np.asarray(sorted(data,key=lambda t:t[2]))
    # extract data according to batch size
    idx = (data[:,0]==lr) & (data[:,1]==batch_size) & (data[:,2] <= 0.92)
    data = data[idx,:]
    return data

#------------------------------------
data = read_data('data/fmnist_momentum.pkl', 10, 0.1)
momentums_fmnist = data[:,2]
sharpness_fmnist = data[:,3]
nonuniformity_fmnist = data[:,4]


data = read_data('data/fmnist_momentum.pkl', 1000, 0.1)
momentums_fmnist_GD = data[:,2]
sharpness_fmnist_GD = data[:,3]
nonuniformity_fmnist_GD = data[:,4]

plt.figure()
plt.plot(momentums_fmnist,nonuniformity_fmnist,'o',markersize=12,lw=5,label=r'FashionMNIST,SGD')
plt.plot(momentums_fmnist_GD,nonuniformity_fmnist_GD,'*',markersize=12,lw=5,label=r'FashionMNIST,GD')
plt.xlabel('Momentum')
plt.ylabel('Non-uniformity')
plt.legend()
plt.savefig('figures/fmnist_momentum_nonuniformity.pdf',bbox_inches='tight')

# # plt.subplot(122)
plt.figure()
plt.plot(momentums_fmnist,sharpness_fmnist,'o',lw=5,markersize=12,label=r'FashionMNIST, SGD')
plt.plot(momentums_fmnist_GD,sharpness_fmnist_GD,'*',lw=5,markersize=12,label=r'FashionMNIST, GD')
plt.xlabel('Momentum')
plt.ylabel('Sharpness')
plt.legend()

plt.savefig('figures/fmnist_momentum_sharpness.pdf',bbox_inches='tight')



#------------------------------------
data = read_data('data/cifar10_momentum.pkl', 10, 0.01)
momentums_cifar = data[:,2]
sharpness_cifar = data[:,3]
nonuniformity_cifar = data[:,4]

data = read_data('data/cifar10_momentum.pkl', 1000, 0.01)
momentums_cifar_GD = data[:,2]
sharpness_cifar_GD = data[:,3]
nonuniformity_cifar_GD = data[:,4]

# plt.figure(figsize=(10,4))
# plt.subplot(121)
plt.figure()
plt.plot(momentums_cifar,nonuniformity_cifar,'o',lw=5,markersize=12,label=r'CIFAR10,SGD')
plt.plot(momentums_cifar_GD,nonuniformity_cifar_GD,'*',lw=5,markersize=12,label=r'CIFAR10,GD')
plt.xlabel('Momentum')
plt.ylabel('Non-uniformity')
plt.legend()
plt.savefig('figures/cifar_momentum_nonuniformity.pdf',bbox_inches='tight')

# plt.subplot(122)
plt.figure()
plt.plot(momentums_cifar,sharpness_cifar,'*',lw=5,markersize=12,label=r'CIFAR10, SGD')
plt.plot(momentums_cifar_GD,sharpness_cifar_GD,'o',lw=5,markersize=12,label=r'CIFAR10, GD')
plt.xlabel('Momentum')
plt.ylabel('Sharpness')
plt.legend()

plt.savefig('figures/cifar_momentum_sharpness.pdf',bbox_inches='tight')