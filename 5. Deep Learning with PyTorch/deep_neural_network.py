# IPython log file

clear
get_ipython().run_line_magic('cls', '')
```
"""  """
#[Out]# '  '
outline = """ """
outline = """ """
outline = """ 1. """
outline = ''' 
1. import the libraries.
2. build three neural network class (sigmoid, tanh, relu)
3. define a function to train the model
4. prepare MNIST datasets for training and validation
5. set the criterion with the loss function
6. create the dataloader for training and validation datasets
7. for each model class:
a. initialize the model with the dimension
b. set the optimizer for that model
c. invoke the training function for that model
8. plot the training loss and the validation accuracy for all the models
'''
outline
#[Out]# ' \n1. import the libraries.\n2. build three neural network class (sigmoid, tanh, relu)\n3. define a function to train the model\n4. prepare MNIST datasets for training and validation\n5. set the criterion with the loss function\n6. create the dataloader for training and validation datasets\n7. for each model class:\na. initialize the model with the dimension\nb. set the optimizer for that model\nc. invoke the training function for that model\n8. plot the training loss and the validation accuracy for all the models\n'
print(outline)
# import the libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(2)
#[Out]# <torch._C.Generator at 0x23f7f45cc50>
# create model class with sigmoid function
class NetSigmoid(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetSigmoid, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmpid(self.linear2(x))
        x = self.linear3(x)
        return x
        
class NetTanh(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmpid(self.linear2(x))
        x = self.linear3(x)
        return x
        
class NetRelu(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
class NetTanh(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x
        
# train the model
# the train function will take the model itself to train it and generate prediction, the criterion to calculate the loss, the trainloader and validation loader as datasets, the optimizer to update the parameters, and the epochs
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {
    'training_loss': [],
    'validation_loss': []
    }
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1, 28*28)
            _, label = torch.max(z, 1)
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {
    'training_loss': [],
    'validation_loss': []
    }
    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
        print(f"Finished training epoch {epoch} - Loss: {loss.data.item()}")
        correct = 0
        print(f"Started validation for epoch {epoch}")
        for x, y in validation_loader:
            z = model(x.view(-1, 28*28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = (correct / len(validation_dataset)) * 100
        useful_stuff['validation_loss'].append(accuracy)
        print(f"Finished validation for epoch {epoch}" - Accuracy: {accuracy})
    return useful_stuff
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {
    'training_loss': [],
    'validation_loss': []
    }
    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.data.item())
        print(f"Finished training epoch {epoch} - Loss: {loss.data.item()}")
        correct = 0
        print(f"Started validation for epoch {epoch}")
        for x, y in validation_loader:
            z = model(x.view(-1, 28*28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = (correct / len(validation_dataset)) * 100
        useful_stuff['validation_loss'].append(accuracy)
        print(f"Finished validation for epoch {epoch} - Accuracy: {accuracy}")
    return useful_stuff
    
# creating datasets, dataloaders, and cross entropy
train_dataset = dsets.MNIST(root='./data', download=True, train=True, shuffle=True, transform=transforms.ToTensor())
train_dataset = dsets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuflle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
criterion = nn.CrossEntropyLoss()
# define dimentions, epochs, learning_rate, initialize models, set optimizers and start training
input_dims = 28*28
hidden_dims1 = 50
hidden_dims2 = 50
output_dims = 10
epochs = 10
learning_rate = 0.01
model_sigmoid = NetSigmoid(input_dims, hidden_dims1, hidden_dims2, output_dims)
optimizer = torch.optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
training_results_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
class NetSigmoid(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetSigmoid, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmpid(self.linear2(x))
        x = self.linear3(x)
        return x
        
training_results_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
class NetSigmoid(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetSigmoid, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
        
training_results_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
class NetSigmoid(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetSigmoid, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)
        return x
        
model_sigmoid = NetSigmoid(input_dims, hidden_dims1, hidden_dims2, output_dims)
training_results_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
model_tanh = NetTanh(input_dims, hidden_dims1, hidden_dims2, output_dims)
optimizer = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
training_results_tanh = train(model_tanh, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
model_relu = Net_Relu(input_dims, hidden_dims1, hidden_dims2, output_dims)
class NetRelu(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
model_relu = Net_Relu(input_dims, hidden_dims1, hidden_dims2, output_dims)
class NetRelu(nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H1)
        self.linear2 = nn.Linear(H1, H2)
        self.linear3 = nn.Linear(H2, D_out)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x
        
model_relu = Net_Relu(input_dims, hidden_dims1, hidden_dims2, output_dims)
model_relu = NetRelu(input_dims, hidden_dims1, hidden_dims2, output_dims)
optimizer = torch.optim.SGD(model_relu.parameters(), lr=learning_rate)
training_results = train(model_relu, criterion, train_loader, validation_loader, optimizer, epochs=epochs)
# analyze the resuls
plt.plot(training_results_sigmoid['training_loss'], label='sigmoid')
#[Out]# [<matplotlib.lines.Line2D at 0x23f19a37380>]
plt.plot(training_results_tanh['training_loss'], label='tanh')
#[Out]# [<matplotlib.lines.Line2D at 0x23f197ccef0>]
plt.plot(training_results_relu['training_loss'], label='relu')
plt.plot(training_results['training_loss'], label='relu')
#[Out]# [<matplotlib.lines.Line2D at 0x23f1990bce0>]
plt.show()
plt.plot(training_results_sigmoid['training_loss'], label='sigmoid')
#[Out]# [<matplotlib.lines.Line2D at 0x23f1a254ef0>]
plt.plot(training_results_tanh['training_loss'], label='tanh')
#[Out]# [<matplotlib.lines.Line2D at 0x23f19ab5550>]
plt.plot(training_results['training_loss'], label='relu')
#[Out]# [<matplotlib.lines.Line2D at 0x23f1990d8e0>]
plt.legend()
#[Out]# <matplotlib.legend.Legend at 0x23f1989fb30>
plt.show()
plt.plot(training_results_sigmoid['validation_accuracy'], label='sigmoid')
plt.plot(training_results_sigmoid['validation_loss'], label='sigmoid')
#[Out]# [<matplotlib.lines.Line2D at 0x23f1a257f50>]
plt.plot(training_results_tanh['validation_loss'], label='tanh')
#[Out]# [<matplotlib.lines.Line2D at 0x23f1a1b10a0>]
plt.plot(training_results['validation_loss'], label='relu')
#[Out]# [<matplotlib.lines.Line2D at 0x23f1a1b27b0>]
plt.legend()
#[Out]# <matplotlib.legend.Legend at 0x23f1bba25a0>
plt.show()
get_ipython().run_line_magic('logstop', '')
