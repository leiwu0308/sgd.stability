import pickle
import math
import json
import numpy as np 
import torch 
from collections import defaultdict

data1 = 'cifar10/cifar10_1.pkl'
data2 = 'cifar10/cifar10_2.pkl'

with open(data1,'rb') as f:
    data1 = pickle.load(f)
with open(data2,'rb') as f:
    data2 = pickle.load(f)

data = sorted(data1 + data2)
res = []
for v in data:
    if math.isnan(v[2]) or v[5] < 80:
        print(v)
    else:
        res.append(v)

with open('cifar10_sgd_collect.pkl','wb') as f:
    pickle.dump(res,f)


# data = res
# res = defaultdict(list)
# for v in data:
#     res[v[0]].append(v[1:])

# for k1,v1 in res.items():
#     tmp = defaultdict(list)
#     for v2 in v1:
#         tmp[v2[0]].append(v2[1:])
#     res[k1] = tmp


# data = [res]


# plt.figure()
# for lr, values in data.items():
#     x_R = values['batch_size']
#     y_R = np.asarray(values['sharpness'])
#     plt.errorbar(np.log10(x_R), y_R[:,0], yerr = y_R[:,1],
#         marker='o',markersize=10, lw=4, label=r'$\eta$=%.2f'%(lr))
# plt.legend(fontsize=18)
# plt.xlabel(r'$\log_{10}$(batch size)',fontsize=35)
# plt.ylabel(r'sharpness',fontsize=35)
# plt.savefig('results/fmnist_batchsize_vs_sharpness.pdf',bbox_inches='tight')

# plt.figure()
# for lr, values in data.items():
#     x_R = values['batch_size']
#     y_R = np.asarray(values['test_accs'])
#     print(y_R[:,0],y_R[:,1])
#     plt.errorbar(np.log10(x_R), y_R[:,0], yerr = y_R[:,1],
#         marker='o',markersize=10, lw=5, label='lr=%.2f'%(lr))
# plt.ylim([78,81.5])
# plt.legend()
# plt.xlabel(r'$\log_{10}$(batch size)',fontsize=35)
# plt.ylabel(r'test accuracy(\%)',fontsize=35)
# plt.savefig('results/fmnist_batchsize_vs_testacc.pdf',bbox_inches='tight')

# plt.figure()
# for lr, values in data.items():
#     x_R = values['batch_size']
#     y_R = np.asarray(values['diversity'])
#     plt.errorbar(np.log10(x_R), y_R[:,0], yerr = y_R[:,1],
#         marker='o',markersize=10, lw=5, label='lr=%.2f'%(lr))
# plt.legend(fontsize=20)
# plt.xlabel(r'$\log_{10}$(batch size)',fontsize=35)
# plt.ylabel(r'nonuniformity',fontsize=35)
# plt.savefig('results/fmnist_batchsize_vs_incoherence.pdf',bbox_inches='tight')