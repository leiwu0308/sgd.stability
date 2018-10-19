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



data = 'data/fmnist/fmnist_sgd_batchsize.pkl'
res = []
with open(data,'rb') as f:
    data = pickle.load(f)
data = sorted(data,key=lambda t:t[5])
data = np.asarray(data)

sharpness = data[:,2]
diversity = data[:,3]
test_acc = data[:,5]

# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.plot(test_acc,sharpness,'o')
# plt.xlabel('Test Accuracy',fontsize=20)
# plt.ylabel('Sharpness',fontsize=20)

# plt.subplot(1,2,2)
plt.plot(diversity,test_acc,'*')
plt.xlabel('Test Accuracy',fontsize=20)
plt.ylabel('Non-uniformity',fontsize=20)
plt.savefig('figures/cifar10_scatter.png',bbox_inches='tight')




