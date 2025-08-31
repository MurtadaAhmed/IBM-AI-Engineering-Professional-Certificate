# IPython log file

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as pl
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
torch.manual_seed(1)
#[Out]# <torch._C.Generator at 0x215df16ffd0>
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layers, layers[1:])])
    def forward(self, x):
        x = x.view(-1, 28*28)
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i < len(self.hidden) - 1:
                x = F.relu(x)
        return x
        
def train(model, train_loader, epochs=10, lr=0.01)"
def train(model, train_loader, epochs=10, lr=0.01)"
def train(model, train_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss)
        print(f"Epoch: {epoch+1}/{epochs}, loss: {losses[-1]:.4f}")
     
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct/total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
layers = [784, 128, 64, 32, 10]
model = Net(layers)
train(model, train_loader, epochs=10, lr=0.01)
evaluate(model, test_loader)
get_ipython().run_line_magic('logstop', '')
