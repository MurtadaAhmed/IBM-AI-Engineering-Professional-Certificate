# IPython log file

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
torch.manual_seed(0)
#[Out]# <torch._C.Generator at 0x2bad666b410>
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return seld.x[index], self.y[index]
    def __len__(self):
        return self.len
        
data_set = Data()
class logistic_regression(nn.Module):
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)
        
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat
        
model = logistic_regression(1)
model.parameters()
#[Out]# <generator object Module.parameters at 0x000002BAD66804A0>
model.state_dict()
#[Out]# OrderedDict([('linear.weight', tensor([[-0.0075]])),
#[Out]#              ('linear.bias', tensor([0.5364]))])
def criterion(yhat,y):
    out = -1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
    return out
    
train_loader = DataLoader(dataset=data_set, batch_size=3)
learning_rate = 2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion_rms(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            get_surface.set_para_loss(model, loss.tolist())
        if epoch % 20 == 0:
            print(epoch, loss)
            
train_model(100)
class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.len
        
data_set = Data()
model = logistic_regression(1)
train_loader = DataLoader(dataset=data_set, batch_size=3)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion_rms(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            get_surface.set_para_loss(model, loss.tolist())
        if epoch % 20 == 0:
            print(epoch, loss)
            
train_model(100)
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            get_surface.set_para_loss(model, loss.tolist())
        if epoch % 20 == 0:
            print(epoch, loss)
            
train_model(100)
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in train_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if epoch % 20 == 0:
            print(epoch, loss)
            
train_model(100)
yhat = model(data_set.x)
yhat
#[Out]# tensor([[8.6126e-09],
#[Out]#         [3.8363e-08],
#[Out]#         [1.7088e-07],
#[Out]#         [7.6113e-07],
#[Out]#         [3.3902e-06],
#[Out]#         [1.5101e-05],
#[Out]#         [6.7259e-05],
#[Out]#         [2.9952e-04],
#[Out]#         [1.3328e-03],
#[Out]#         [5.9092e-03],
#[Out]#         [2.5795e-02],
#[Out]#         [1.0550e-01],
#[Out]#         [3.4440e-01],
#[Out]#         [7.0059e-01],
#[Out]#         [9.1245e-01],
#[Out]#         [9.7891e-01],
#[Out]#         [9.9519e-01],
#[Out]#         [9.9892e-01],
#[Out]#         [9.9976e-01],
#[Out]#         [9.9995e-01]], grad_fn=<SigmoidBackward0>)
labels = yhat > 0.5
labels
#[Out]# tensor([[False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [False],
#[Out]#         [ True],
#[Out]#         [ True],
#[Out]#         [ True],
#[Out]#         [ True],
#[Out]#         [ True],
#[Out]#         [ True],
#[Out]#         [ True]])
data_set.y
#[Out]# tensor([[0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.]])
data_set.y.type()
#[Out]# 'torch.FloatTensor'
data_set.y.type(torch.ByteTensor)
#[Out]# tensor([[0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1]], dtype=torch.uint8)
labels == data_set.y.type(torch.ByteTensor)
#[Out]# tensor([[True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True]])
torch.mean(labels == data_set.y.type(torch.ByteTensor))
torch.mean(labels == data_set.y.type(torch.ByteTensor).type(torch.flo))
torch.mean(labels == data_set.y.type(torch.ByteTensor).type(torch.float))
data_set.y.type(torch.ByteTensor).type(torch.float)
#[Out]# tensor([[0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.]])
results_coverted = (labels = data_set.y.type(torch.ByteTensor)).type(torch.float)
results = labels = data_set.y.type(torch.ByteTensor)
results = results.type()
results = labels = data_set.y.type(torch.ByteTensor)
results.type()
#[Out]# 'torch.ByteTensor'
results
#[Out]# tensor([[0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [0],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1],
#[Out]#         [1]], dtype=torch.uint8)
results = results.type(torch.float())
results = results.type(torch.float)
results
#[Out]# tensor([[0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [0.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.],
#[Out]#         [1.]])
torch.mean(results)
#[Out]# tensor(0.3500)
results = labels == data_set.y.type(torch.ByteTensor)
results
#[Out]# tensor([[True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True],
#[Out]#         [True]])
results = results.type(torch.float)
torch.mean(results)
#[Out]# tensor(1.)
get_ipython().run_line_magic('logstop', '')
