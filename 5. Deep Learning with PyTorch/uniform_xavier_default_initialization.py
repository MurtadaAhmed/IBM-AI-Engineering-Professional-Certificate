# IPython log file

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
torch.manual_seed(0)
#[Out]# <torch._C.Generator at 0x21a2d410030>
train_dataset = dsets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())
test_dataset = dsets.MNIST(root='./data', download=True, train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
class Net(nn.Module):
    def __init__(self, layers, init_type='default')
class Net(nn.Module):
    def __init__(self, layers, init_type='default'):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layers, layers[1:]))
class Net(nn.Module):
    def __init__(self, layers, init_type='default'):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layers, layers[1:])])
        for layer in self.hidden:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(layer.weight_
            elif init_type == 'uniform'"
class Net(nn.Module):
    def __init__(self, layers, init_type='default'):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layers, layers[1:])])
        for layer in self.hidden:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(layer.weight_
            elif init_type == 'uniform':
class Net(nn.Module):
    def __init__(self, layers, init_type='default'):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layers, layers[1:])])
        for layer in self.hidden:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == 'uniform':
                layer.weight.data.uniform_(0, 1)
    def forward(self, x):
        x = x.view(-1, 28*28)
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i < len(self.hidden) - 1:
                x = torch.tang(x)
        return x
        
def train(model, train_loader, test_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEnropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_accuracies = []
    
def train(model, train_loader, test_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEnropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracies.append(100 * correct/total)
        print(f"Epoch {epoch + 1}/{eochs}, Loss: {train_losses[-1]:.2f}, Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies
    
layers = [784, 100, 10]
init_types = ['default', 'xavier', 'uniform']
results = {}
for init in init_types:
    print(f'\nTraining with {init} initialization')
    model = Net(layers, init_type=init)losses, accuracies = train(model, train_loader, test_loader, epochs=10, lr=0.01)
    results[init] = {'losses': losses, 'accuracies': accuracies}
results = {}
for init in init_types:
    print(f'\nTraining with {init} initialization')
    model = Net(layers, init_type=init)
    losses, accuracies = train(model, train_loader, test_loader, epochs=10, lr=0.01)
    results[init] = {'losses': losses, 'accuracies': accuracies}
    
def train(model, train_loader, test_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEnropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracies.append(100 * correct/total)
        print(f"Epoch {epoch + 1}/{eochs}, Loss: {train_losses[-1]:.2f}, Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies
    
results = {}
for init in init_types:
    print(f'\nTraining with {init} initialization')
    model = Net(layers, init_type=init)
    losses, accuracies = train(model, train_loader, test_loader, epochs=10, lr=0.01)
    results[init] = {'losses': losses, 'accuracies': accuracies}
    
def train(model, train_loader, test_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracies.append(100 * correct/total)
        print(f"Epoch {epoch + 1}/{eochs}, Loss: {train_losses[-1]:.2f}, Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies
    
results = {}
for init in init_types:
    print(f'\nTraining with {init} initialization')
    model = Net(layers, init_type=init)
    losses, accuracies = train(model, train_loader, test_loader, epochs=10, lr=0.01)
    results[init] = {'losses': losses, 'accuracies': accuracies}
    
class Net(nn.Module):
    def __init__(self, layers, init_type='default'):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList([nn.Linear(in_size, out_size) for in_size, out_size in zip(layers, layers[1:])])
        for layer in self.hidden:
            if init_type == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif init_type == 'uniform':
                layer.weight.data.uniform_(0, 1)
    def forward(self, x):
        x = x.view(-1, 28*28)
        for i, layer in enumerate(self.hidden):
            x = layer(x)
            if i < len(self.hidden) - 1:
                x = torch.tanh(x)
        return x
        
results = {}
for init in init_types:
    print(f'\nTraining with {init} initialization')
    model = Net(layers, init_type=init)
    losses, accuracies = train(model, train_loader, test_loader, epochs=10, lr=0.01)
    results[init] = {'losses': losses, 'accuracies': accuracies}
    
def train(model, train_loader, test_loader, epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accuracies.append(100 * correct/total)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.2f}, Test Accuracy: {test_accuracies[-1]:.2f}%")
    return train_losses, test_accuracies
    
results = {}
for init in init_types:
    print(f'\nTraining with {init} initialization')
    model = Net(layers, init_type=init)
    losses, accuracies = train(model, train_loader, test_loader, epochs=10, lr=0.01)
    results[init] = {'losses': losses, 'accuracies': accuracies}
    
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for init in init_types:
    plt.plot(results[init]['losses'], label=f'{init} Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
for init in init_types:
    plt.plot(results[init]['accuracies'], label=f'{init} Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Test Accuracy')
plt.tight_layout()
plt.show()
get_ipython().run_line_magic('logstop', '')
