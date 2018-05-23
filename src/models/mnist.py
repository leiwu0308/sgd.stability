import torch
import torch.nn as nn
import torch.nn.functional as F

class ResFNN(nn.Module):
    def __init__(self,depth=2,width=200,input_dim=784,skip_connection=False):
        super(ResFNN,self).__init__()
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self.skip_connection = skip_connection
        if self.depth > 0:
            self.fc = [nn.Linear(input_dim,width)]\
                    + [nn.Linear(width,width) for _ in range(depth-1)]\
                    + [nn.Linear(width,10)]
        else:
            self.fc = [nn.Linear(input_dim,10)]
        self.fc = nn.ModuleList(self.fc)

    def forward(self,x):
        o = x.view(x.size(0),-1)
        for i,m in enumerate(self.fc):
            o_pre = o
            o = m(o)
            if i < len(self.fc) -1:
                o = F.relu(o)
                if i>0 and self.skip_connection:
                    o += o_pre
        return o


#====================================
# LeNet
#====================================
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1) # 28-5+1=24
        self.conv2 = nn.Conv2d(6,16,5,stride=1) # 12-5+1=8
        self.fc1 = nn.Linear(4*4*16,200)
        self.fc2 = nn.Linear(200,10)

    def forward(self,x):
        if x.ndimension()==3:
            x = x.unsqueeze(0)
        o = F.relu(self.conv1(x))
        o = F.avg_pool2d(o,2,2)

        o = F.relu(self.conv2(o))
        o = F.avg_pool2d(o,2,2)

        o = o.view(o.shape[0],-1)
        o = self.fc1(o)
        o = F.relu(o)
        o = self.fc2(o)
        return o

#====================================
# API
#====================================
def lenet(depth=-1):
    return LeNet()

def resfnn(depth,width=200,skip_connection=False):
    return ResFNN(depth=depth,width=width,skip_connection=skip_connection)


