import time
import random
import numpy as np
import torch
import torchvision.datasets as dsets

###############################
# High-level function to load data
###############################
def load_mnist(batch_size = 100,num_wrong = 0,nsamples = -1,
            fashion = False,root = None,stop = True, one_hot=False):
    train_loader = MNISTLoader(
                            root = root,
                            train = True,
                            batch_size = batch_size,
                            fashion = fashion,
                            nsamples = nsamples,
                            num_wrong = num_wrong,
                            stop = stop,
                            one_hot = one_hot
                        )
    test_loader = MNISTLoader(
                            root = root,
                            train = False,
                            batch_size = batch_size,
                            fashion = fashion,
                            num_wrong = num_wrong,
                            stop=True,
                            one_hot = one_hot
                        )

    return train_loader,test_loader

def load_cifar(batch_size=100,num_wrong=0,nsamples=-1, nclasses=10,
                        cat=10,root=None,stop=True,one_hot=False):
    train_loader = CIFARLoader(
                            root = root,
                            cat = cat,
                            batch_size = batch_size,
                            train = True,
                            nsamples = nsamples,
                            num_wrong = num_wrong,
                            stop = stop,
                            one_hot = one_hot,
                            nclasses = nclasses
                        )
    test_loader = CIFARLoader(
                            root = root,
                            cat = cat,
                            batch_size = batch_size,
                            train = False,
                            stop = True,
                            one_hot = one_hot,
                            nclasses = nclasses
                        )
    return train_loader,test_loader




###############################
# low-level api
###############################
def select_samples(X,y,nsamples=-1,num_wrong=0):
    np.random.seed(12345)
    if nsamples <= 0:
        nsamples = y.shape[0]
    n_classes = y.max()
    total_samples = min(nsamples + num_wrong,y.shape[0])
    X_new = X[0:total_samples].clone()
    y_new = y[0:total_samples].clone()

    for i in range(nsamples,nsamples+num_wrong):
        y_true = y_new[i]
        while True:
            y_wrong = np.random.randint(0,n_classes)
            if y_wrong != y_true:
                y_new[i] = y_wrong
                break
    return X_new,y_new


class RandomHorizontalFlip:
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self,x):
        if torch.rand(1)[0] < self.p:
            x = torch.from_numpy(x.numpy()[:,:,:,::-1].copy())
        return x


class CIFARLoader:
    def __init__(self,train=True,batch_size=100,
            transform=None,padding=0,root=None,
            nsamples=-1,cat=10,num_wrong=0,
            nclasses =2, stop=True,one_hot=False):
        self.cat = cat
        self.train = train
        self.batch_size = batch_size
        self.padding = padding
        self.transform = transform
        self.stop = stop

        if cat == 10:
            self.root = './data/cifar10' if root is None else root
            cifar_loader = dsets.CIFAR10
        else:
            self.root = './data/cifar100' if root is None else root
            cifar_loader = dsets.CIFAR100


        # load data
        if train:
            dset = cifar_loader(self.root,train=True,download=True)
            X,y = dset.train_data,dset.train_labels
        else:
            dset = cifar_loader(self.root,train=False,download=True)
            X,y = dset.test_data,dset.test_labels

        # process data
        X = torch.from_numpy(X.transpose([0,3,1,2])).float()/255
        y = torch.LongTensor(y)

        if nclasses == 2:
            X_new = torch.Tensor(10000,3,32,32)
            y_new = torch.LongTensor(10000)
            idx = 0
            for i in range(len(y)):
                if y[i] == 0 or y[i] == 1:
                    y_new[idx] = y[i]
                    X_new[idx,:,:,:] = X[i,:,:,:]
                    idx += 1
            X = X_new[0:idx,:,:,:]
            y = y_new[0:idx]

        if self.train:
            self.X,self.y = select_samples(X,y,nsamples,num_wrong)
        else:
            self.X,self.y = X,y

        self.idx = 0
        self.nsamples = len(self.y)

        if one_hot:
            self.y = to_one_hot(self.y)

        if padding > 0 and train:
            N,C,W,H = self.X.shape
            X_new = torch.zeros(N,C,W+2*padding,H+2*padding)
            X_new[:,:,padding:W+padding,padding:H+padding] = self.X
            self.X = X_new
            self.W = W
            self.H = H

    def __len__(self):
        length = self.nsamples // self.batch_size
        if self.nsamples > length * self.batch_size:
            length += 1
        return length

    def __iter__(self):
        return self

    def reset(self,batch_size=None,stop=True):
        self.idx = 0
        if batch_size is not None:
            self.batch_size = batch_size
        self.stop = stop

    def __next__(self):
        if self.idx >= self.nsamples:
            self.idx = 0
            rnd_idx = torch.randperm(self.nsamples)
            self.X = self.X[rnd_idx]
            self.y = self.y[rnd_idx]
            if self.stop:
                raise StopIteration

        idx_end = min(self.idx+self.batch_size,self.nsamples)
        batch_X = self.X[self.idx:idx_end]
        batch_y = self.y[self.idx:idx_end]
        self.idx = idx_end

        if self.transform:
            batch_X = self.transform(batch_X)
        if self.padding>0 and self.train:
            h_off = random.randint(0,2*self.padding)
            w_off = random.randint(0,2*self.padding)
            batch_X = batch_X[:,:,w_off:w_off+self.W,h_off:h_off+self.H]
        return batch_X,batch_y




class MNISTLoader:
    def __init__(self,train=True,batch_size=100,
                    root=None,fashion=False,
                    nsamples=-1,num_wrong=0,stop=True,one_hot=False):
        self.train = train
        self.fashion = fashion
        self.stop = stop
        self.batch_size = batch_size

        if not fashion:
            self.root = './data/mnist' if root is None else root
            dset_loader = dsets.MNIST
        else:
            self.root = './data/fashionmnist' if root is None else root
            dset_loader = dsets.FashionMNIST
        if train:
            dset = dset_loader(self.root,train=True,download=True)
            X,y = dset.train_data,dset.train_labels
            X,y = select_samples(X,y,nsamples,num_wrong)
        else:
            dset = dset_loader(self.root,train=False,download=True)
            X,y = dset.test_data,dset.test_labels

        self.X = X.unsqueeze(1).float()/255
        self.y = y
        self.idx = 0
        self.nsamples = len(y)

        if one_hot:
            self.y = to_one_hot(self.y)

    def __len__(self):
        length = self.nsamples // self.batch_size
        if self.nsamples > length * self.batch_size:
            length += 1
        return length

    def __iter__(self):
        return self

    def reset(self,batch_size=None,stop=True):
        self.idx = 0
        if batch_size is not None:
            self.batch_size = batch_size
        self.stop = stop

    def __next__(self):
        if self.idx >= self.nsamples:
            self.idx = 0
            rnd_idx = torch.randperm(self.nsamples)
            self.X = self.X[rnd_idx]
            self.y = self.y[rnd_idx]
            if self.stop:
                raise StopIteration

        idx_end = min(self.idx+self.batch_size,self.nsamples)
        batch_X = self.X[self.idx:idx_end]
        batch_y = self.y[self.idx:idx_end]
        self.idx = idx_end

        return batch_X,batch_y

def to_one_hot(labels):
    if labels.ndimension()==1:
        labels.unsqueeze_(1)
    n_samples = labels.shape[0]
    n_classes = labels.max()+1
    one_hot_labels = torch.FloatTensor(n_samples,n_classes)
    one_hot_labels.zero_()
    one_hot_labels.scatter_(1, labels, 1)
    return one_hot_labels


