import os
import pickle
import argparse
import json
from collections import defaultdict
import numpy as np
import matplotlib as mpl 

mpl.use('Agg')
mpl.rc('text',usetex=True)
mpl.rcParams['legend.fontsize']=18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18
import matplotlib.pyplot as plt

from ipython.plot import smooth_plot


def read_data(filename1,filename2):
    with open(filename1,'rb') as f:
        sharpness = pickle.load(f)
    with open(filename2,'rb') as f:
        loss = pickle.load(f)

    iters = [t['iter'] for t in sharpness['traj']]
    sharpness = [t['info'] for t in sharpness['traj']]
    train_acc = [t['info'][1] for t in loss['train']]
    train_loss = [t['info'][0] for t in loss['train']]
    test_acc = [t['info'][1] for t in loss['test']]
    test_loss = [t['info'][0] for t in loss['test']]

    return np.asarray([iters,sharpness,train_acc,train_loss,test_acc,test_loss])

def restart2():
    data = 'results/data/fmnist/restart_n1000_w200/lr0.1_bz100_sharpness.pkl'
    with open(data,'rb') as f:
        sharpness = pickle.load(f)
    data = 'results/data/fmnist/restart_n1000_w200/lr0.1_bz100_loss.pkl'
    with open(data,'rb') as f:
        loss = pickle.load(f)
    iters1 = [t['iter'] for t in sharpness['traj']]
    sharpness1 = [t['info'] for t in sharpness['traj']]
    train_acc1 = [t['info'][1] for t in loss['train']]
    train_loss1 = [t['info'][0] for t in loss['train']]
    test_acc1 = [t['info'][1] for t in loss['test']]
    test_loss1 = [t['info'][0] for t in loss['test']]

    data = 'results/data/fmnist/restart_n1000_w200/lr0.1_bz4_sharpness.pkl'
    with open(data,'rb') as f:
        sharpness = pickle.load(f)
    data = 'results/data/fmnist/restart_n1000_w200/lr0.1_bz4_loss.pkl'
    with open(data,'rb') as f:
        loss = pickle.load(f)

    iters2 = [t['iter'] for t in sharpness['traj']]
    sharpness2 = [t['info'] for t in sharpness['traj']]
    train_acc2 = [t['info'][1] for t in loss['train']]
    train_loss2 = [t['info'][0] for t in loss['train']]
    test_acc2 = [t['info'][1] for t in loss['test']]
    test_loss2 = [t['info'][0] for t in loss['test']]

    data = 'results/data/fmnist/restart_n1000_w200/lr0.3_bz1200_sharpness.pkl'
    with open(data,'rb') as f:
        sharpness = pickle.load(f)
    data = 'results/data/fmnist/restart_n1000_w200/lr0.3_bz1200_loss.pkl'
    with open(data,'rb') as f:
        loss = pickle.load(f)
    iters3 = [t['iter'] for t in sharpness['traj']]
    sharpness3 = [t['info'] for t in sharpness['traj']]
    train_acc3 = [t['info'][1] for t in loss['train']]
    train_loss3 = [t['info'][0] for t in loss['train']]
    test_acc3 = [t['info'][1] for t in loss['test']]
    test_loss3 = [t['info'][0] for t in loss['test']]


    data = 'results/data/fmnist/restart_n1000_w200/lr0.5_bz1200_sharpness.pkl'
    with open(data,'rb') as f:
        sharpness = pickle.load(f)
    data = 'results/data/fmnist/restart_n1000_w200/lr0.5_bz1200_loss.pkl'
    with open(data,'rb') as f:
        loss = pickle.load(f)
    iters4 = [t['iter'] for t in sharpness['traj']]
    sharpness4 = [t['info'] for t in sharpness['traj']]
    train_acc4 = [t['info'][1] for t in loss['train']]
    train_loss4 = [t['info'][0] for t in loss['train']]
    test_acc4 = [t['info'][1] for t in loss['test']]
    test_loss4 = [t['info'][0] for t in loss['test']]

    iter_max = 1500
    plt.figure()
    plt.plot(iters1,sharpness1,'-',lw=4)
    plt.plot(iters2,sharpness2,'-',lw=4)
    plt.plot(iters3,sharpness3,'-',lw=4)
    plt.plot(iters4,sharpness4,'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel('sharpness',fontsize=30)
    plt.xlim([-0.5,iter_max])
    plt.savefig('results/fmnist_n1000_w200_restart_sharpness.pdf',bbox_inches='tight')


    plt.figure()
    plt.plot(iters1,train_acc1,'-',lw=4,label=r'$\eta$=0.1,B=100, test acc: %.2f'%(test_acc1[-1]))
    plt.plot(iters2,train_acc2,'-',lw=4,
                label=r'$\eta$=0.1,B=4, test acc: %.2f'%(test_acc2[-1]))
    plt.plot(iters3,train_acc3,'-',lw=4,
                label=r'$\eta$=0.3,B=1200, test acc: %.2f'%(test_acc3[-1]))
    plt.plot(iters4,train_acc4,'-',lw=4,
                label=r'$\eta$=0.5,B=1200, test acc: %.2f'%(test_acc4[-1]))
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'training accuracy',fontsize=30)
    plt.legend(fontsize=19)
    plt.xlim([-0.5,iter_max])
    plt.savefig('results/fmnist_n1000_w200_restart_trainacc.pdf',bbox_inches='tight')

    plt.figure()
    plt.plot(iters1,test_acc1,'-',lw=4)
    plt.plot(iters2,test_acc2,'-',lw=4)
    plt.plot(iters3,test_acc3,'-',lw=4)
    plt.plot(iters4,test_acc4,'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'test accuracy',fontsize=30) 
    plt.xlim([-0.5,iter_max])

    baseline = np.ones(len(iters1))*test_acc1[0]
    plt.plot(iters1,baseline,'--k',lw=3)

    plt.savefig('results/fmnist_n1000_w200_restart_testacc.pdf',bbox_inches='tight')

def restart_cifar(x_max=1000):
    data1 = read_data('results/data/cifar10/cifar_restart/lr0.01_bz1_sharpness.pkl', 
                        'results/data/cifar10/cifar_restart/lr0.01_bz1_loss.pkl')
    data2 = read_data('results/data/cifar10/cifar_restart/lr0.01_bz4_sharpness.pkl', 
                        'results/data/cifar10/cifar_restart/lr0.01_bz4_loss.pkl')
    data3 = read_data('results/data/cifar10/cifar_restart/lr0.01_bz25_sharpness.pkl',
                        'results/data/cifar10/cifar_restart/lr0.01_bz25_loss.pkl')
    # data4 = read_data('results/data/cifar10/cifar_restart/lr0.1_bz1000_sharpness.pkl',
    #                     'results/data/cifar10/cifar_restart/lr0.1_bz1000_loss.pkl')
    data5 = read_data('results/data/cifar10/cifar_restart/lr0.2_bz1000_sharpness.pkl',
                        'results/data/cifar10/cifar_restart/lr0.2_bz1000_loss.pkl')

    data_list = [data1,data2,data5]
    labels = [r'$\eta$=0.01,B=1', r'$\eta$=0.01,B=4',
             r'$\eta$=0.2,B=1000']

    plt.figure()
    for data,label in zip(data_list,labels):
        plt.plot(data[0],data[1],'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel('sharpness',fontsize=30)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/cifar10_restart_sharpness.pdf',bbox_inches='tight')


    plt.figure()
    for data in data_list:
        plt.plot(data[0],data[2],'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'training accuracy',fontsize=30)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/cifar10_restart_trainacc.pdf',bbox_inches='tight')

    plt.figure()
    for data,label in zip(data_list,labels):
        plt.plot(data[0],data[4],'-',lw=4,label=r'%s, test acc: %.2f'%(label,data[4][-1]))
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'test accuracy',fontsize=30)
    plt.legend(loc=4)

    baseline = np.ones(len(data1[0]))*data1[4][0]
    plt.plot(data1[0],baseline,'--k',lw=3)
    # plt.legend(fontsize=20)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/cifar10_restart_testacc.pdf',bbox_inches='tight')

def restart_lbfgs(x_max=1000):
    data1 = read_data('results/data/fmnist/lbfgs_lr0.8_bz1200_sharpness.pkl',
                        'results/data/fmnist/lbfgs_lr0.8_bz1200_loss.pkl')
    data2 = read_data('results/data/fmnist/lbfgs_lr0.4_bz0.4_sharpness.pkl',
                        'results/data/fmnist/lbfgs_lr0.4_bz0.4_loss.pkl')


    data_list = [data1,data2]
    labels = [r'GD $\eta$=0.8', r'SGD $\eta$=0.4,B=4']

    plt.figure()
    for data,label in zip(data_list,labels):
        print(len(data))
        plt.plot(data[0],data[1],'-',lw=4,label=r'%s, test acc: %.2f'%(label,data[4][-1]))
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel('sharpness',fontsize=30)
    plt.legend()
    plt.xlim([-0.5,x_max])
    plt.savefig('results/fmnist_lbfgs_restart_sharpness.pdf',bbox_inches='tight')


    plt.figure()
    for data in data_list:
        plt.plot(data[0],data[2],'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'training accuracy',fontsize=30)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/fmnist_lbfgs_restart_trainacc.pdf',bbox_inches='tight')

    plt.figure()
    for data in data_list:
        plt.plot(data[0],data[4],'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'test accuracy',fontsize=30)

    baseline = np.ones(len(data1[0]))*data1[4][0]
    plt.plot(data1[0],baseline,'--k',lw=3)
    # plt.legend(fontsize=20)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/fmnist_lbfgs_restart_testacc.pdf',bbox_inches='tight')

def restart_n1000_w0(x_max=1000):
    data1 = read_data('results/data/fmnist/restart_n1000_w0/lr0.1_bz1_sharpness.pkl',
                        'results/data/fmnist/restart_n1000_w0/lr0.1_bz1_loss.pkl')

    data2 = read_data('results/data/fmnist/restart_n1000_w0/lr0.1_bz4_sharpness.pkl',
                        'results/data/fmnist/restart_n1000_w0/lr0.1_bz4_loss.pkl')


    data3 = read_data('results/data/fmnist/restart_n1000_w0/lr0.3_bz1000_sharpness.pkl',
                        'results/data/fmnist/restart_n1000_w0/lr0.3_bz1000_loss.pkl')

    data4 = read_data('results/data/fmnist/restart_n1000_w0/lr0.5_bz1000_sharpness.pkl', 
                        'results/data/fmnist/restart_n1000_w0/lr0.5_bz1000_loss.pkl')


    data_list = [data1,data2,data3,data4]
    labels = [r'$\eta$=0.1,B=1', r'$\eta$=0.1,B=4', r'$\eta$=0.3,B=1000', r'$\eta$=0.5,B=1000']

    plt.figure()
    for data,label in zip(data_list,labels):
        print(len(data))
        plt.plot(data[0],data[1],'-',lw=4,label=r'%s, test acc: %.2f'%(label,data[4][-1]))
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel('sharpness',fontsize=30)
    plt.legend()
    plt.xlim([-0.5,x_max])
    plt.savefig('results/fmnist_n1000_w0_restart_sharpness.pdf',bbox_inches='tight')


    plt.figure()
    for data in data_list:
        plt.plot(data[0],data[2],'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'training accuracy',fontsize=30)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/fmnist_n1000_w0_restart_trainacc.pdf',bbox_inches='tight')

    plt.figure()
    for data in data_list:
        plt.plot(data[0],data[4],'-',lw=4)
    plt.xlabel(r'\# iteration',fontsize=30)
    plt.ylabel(r'test accuracy',fontsize=30)

    baseline = np.ones(len(data1[0]))*data1[4][0]
    plt.plot(data1[0],baseline,'--k',lw=3)
    # plt.legend(fontsize=20)
    plt.xlim([-0.5,x_max])
    plt.savefig('results/fmnist_n1000_w0_restart_testacc.pdf',bbox_inches='tight')

if __name__ == '__main__':
    # restart_n1000_w0(x_max=3000)
    # restart2()
    # restart_cifar(x_max=15000)
    restart_lbfgs(5000)