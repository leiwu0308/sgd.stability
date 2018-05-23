import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

def get_cifar_mean_std():
    mean = torch.Tensor([x/255 for x in [125.3,123.0, 113.9]]).view(1,3,1,1)
    std  = torch.Tensor([x/255 for x in [63.0, 62.1, 66.7]]).view(1,3,1,1)
    mean_var = Variable(mean.cuda())
    std_var = Variable(std.cuda())
    return mean_var,std_var

#====================================
# FNN
#====================================
class FNN(nn.Module):
    def __init__(self,W=500,D=3):
        super(FNN,self).__init__()
        fc = [nn.Linear(3*32*32,W)]
        for i in range(D-1):
            fc.append(nn.Linear(W,W))
        fc.append(nn.Linear(W,10))
        self.fc = nn.ModuleList(fc)
        self.D = D
        self.name = 'fnn'

        self.mean_var, self.std_var = get_cifar_mean_std()

    def forward(self,x):
        x = (x-self.mean_var)/self.std_var
        o = x.view(x.size(0),-1)
        for i in range(self.D+1):
            o = self.fc[i](o)
            if i < self.D:
                o = F.relu(o)
        return o

#====================================
# LeNet
#====================================
class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(3,16,5,stride=1) # 32-5+1=28
        self.conv2 = nn.Conv2d(16,32,5,stride=1) # 14-5+1=10
        self.conv3 = nn.Conv2d(32,128,5,stride=1) # 5-5+1 = 1
        self.fc1 = nn.Linear(128,500)
        self.fc2 = nn.Linear(500,num_classes)

    def forward(self,x):
        if x.ndimension()==3:
            x = x.unsqueeze(0)
        o = self.conv1(x)
        o = F.relu(o)
        o = F.avg_pool2d(o,2,2)

        o = self.conv2(o)
        o = F.relu(o)
        o = F.avg_pool2d(o,2,2)

        o = self.conv3(o)

        o = o.view(o.shape[0],-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o

#====================================
# ResNet
#====================================
def conv3x3(in_channels,out_channels,stride=1):
    return nn.Conv2d(in_channels,out_channels,kernel_size=3,
                    stride=stride,padding=1,bias=True)


class short_cut(nn.Module):
    def __init__(self,in_channels,out_channels,type='A'):
        super(short_cut,self).__init__()
        self.type = 'D' if in_channels == out_channels else type
        if self.type == 'C':
            self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,padding=0,stride=2,bias=False)
            self.bn   = nn.BatchNorm2d(out_channels)
        elif self.type == 'A':
            self.avg  = nn.AvgPool2d(kernel_size=1,stride=2)

    def forward(self,x):
        if self.type == 'A':
            x = self.avg(x)
            return torch.cat((x,x.mul(0)),1)
        elif self.type == 'C':
            x = self.conv(x)
            x = self.bn(x)
            return x
        elif self.type == 'D':
            return x

class residual_block(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,shortcutType='D'):
        super(residual_block,self).__init__()
        self.conv1 = conv3x3(in_channels,out_channels,stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels,out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = short_cut(in_channels,out_channels,type=shortcutType)

    def forward(self,x):
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)
        o = self.conv2(o)
        o = self.bn2(o)
        o += self.shortcut(x)

        o = self.relu(o)
        return o

class ResNet(nn.Module):
    def __init__(self,block,depth,multiple=2,num_classes=10,shortcutType='A'):
        super(ResNet,self).__init__()
        assert (depth-2) %6 == 0 , 'depth should be 6*m + 2, like 20 32 44 56 110'
        num_blocks = (depth-2)//6
        print('resnet: depth: %d, # of blocks at each stage: %d'%(depth,num_blocks))

        self.in_channels = 8*multiple
        self.conv = conv3x3(3,self.in_channels)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.stage1 = self._make_layer(block,8*multiple,num_blocks,1) # 32x32x16
        self.stage2 = self._make_layer(block,16*multiple,num_blocks,2) # 16x16x32
        self.stage3 = self._make_layer(block,32*multiple,num_blocks,2) # 8x8x64
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(32*multiple,num_classes)
        self.name = 'resnet'

        # initialization by Kaiming strategy
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                fin = m.kernel_size[0]*m.kernel_size[1]*m.out_channels #??????
                m.weight.data.normal_(0,math.sqrt(2.0/fin))
                #m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

        self.mean_var,self.std_var = get_cifar_mean_std()


    def forward(self,x):
        x = (x-self.mean_var)/self.std_var
        o = self.conv(x)
        o = self.bn(o)
        o = self.relu(o)

        o = self.stage1(o)
        o = self.stage2(o)
        o = self.stage3(o)

        o = self.avg_pool(o)
        o = o.view(o.size(0),-1)
        o = self.fc(o)
        return o

    def _make_layer(self,block,out_channels,num_blocks,stride=1,shortcutType='A'):
        layers = []
        layers.append(block(self.in_channels,out_channels,stride,shortcutType=shortcutType))
        self.in_channels = out_channels
        for i in range(1,num_blocks):
            layers.append(block(out_channels,out_channels))
        return nn.Sequential(*layers)


#====================================
# API
#====================================
def fnn(width,depth):
    return FNN(width,depth)

def lenet(num_classes=10):
    return LeNet(num_classes)

def resnet(width=2,depth=14,num_classes=10):
    if (depth-2)%6 != 0:
        raise ValueError('depth: %d is not legal depth for resnet'%(depth))
    return ResNet(residual_block,depth,width,num_classes)
