import os
import pickle
import argparse
import json
from collections import defaultdict
import numpy as np
import matplotlib as mpl 

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=20
mpl.rcParams['xtick.labelsize'] = 25
mpl.rcParams['ytick.labelsize'] = 25
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set()



# Scatter Plot

# data = 'results/data/scatter.pkl'
# with open(data,'rb') as f:
#     data = pickle.load(f)
#     data = np.asarray(data)
# sharpness = data[:,2]
# nonuniformity = data[:,3]
# test_acc = data[:,5]
# plt.figure()
# plt.plot(sharpness,nonuniformity,'*k',markersize=15)
# plt.xlabel('sharpness',fontsize=35)
# plt.ylabel('nonuniformity',fontsize=35)
# plt.savefig('results/fmnist_n1000_sharpness_nonuniformity.pdf',bbox_inches='tight')


data = 'results/data/cifar10/cifar10_sgd_collect.pkl'
res = []
with open(data,'rb') as f:
    data = pickle.load(f)

for v in data:
    if v[0] <= 0.05:
        res.append(v)
        print('%.1e\t %.1e\t %.2f\t %.2e'%(v[2],v[3],v[5],v[6]))
data = res
data = np.asarray(data)

sharpness = data[:,2]
diversity = data[:,3]
test_acc = data[:,5]

plt.figure()
plt.plot(sharpness,test_acc,'*k',markersize=15)
plt.xlabel('sharpness',fontsize=35)
plt.ylabel('test accuracy',fontsize=35)
plt.savefig('results/cifar10_sharp_vs_testacc',bbox_inches='tight')

plt.figure()
plt.plot(diversity,test_acc,'*k',markersize=15)
plt.xlabel('nonuniformity',fontsize=35)
plt.ylabel('test accuracy',fontsize=35)
plt.savefig('results/cifar10_nonuniformity_vs_testacc',bbox_inches='tight')

#=====================================================
# data4 = 'results/data/mnist_gd_samples.pkl'
# with open(data4,'rb') as f:
#     data4 = pickle.load(f)
#     data4 = np.asarray(data4)
# res = defaultdict(list)
# for v in data4:
#     lr = v[0]
#     res[lr].append(list(v[1:]))

# x_R = []
# y_R = []
# e_R = []
# for k,v in sorted(res.items()):
#     x_R.append(k)
#     v = np.asarray(v)
#     mean = np.mean(v[:,1])
#     std = np.std(v[:,1])
#     y_R.append(mean)
#     e_R.append(std)
# print(x_R)
# print(y_R)
# print(e_R)
# print(2/np.asarray(x_R))

# plt.figure()
# plt.errorbar(np.log10(x_R),np.log10(y_R),yerr=np.log10(e_R),lw=4,marker='o',label='gradient descent')
# plt.xlabel('learning rate',fontsize=20)
# plt.ylabel('sharpness',fontsize=20)
# plt.legend(fontsize=20)
# plt.savefig('mnist_gd_sharpness.pdf',bbox_inches='tight')


