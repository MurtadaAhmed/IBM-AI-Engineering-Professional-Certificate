# IPython log file

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn as nn
class Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = self.linear2(x)
        return x
        
class NetTanh(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x
        
class NetTanh(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NetTanh, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x = torch.tanh(self.linear1(x))
        x = self.linear2(x)
        return x
        
class NetRelu(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(NetRelu, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x
        
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {
    'training_loss': [],
    'validation_accuracy': [],
    }
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer = optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.item())
        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1,28*28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = (correct / len(validation_dataset)) * 100
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff
    
            
train_dataset = dsets.MNIST(root='./data', download=True, train=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data', download=True, transform=transforms.ToTensor())
validation_dataset = dsets.MNIST(root='./data',train=False, download=True, transform=transforms.ToTensor())
criterion = nn.CrossEntropyLoss()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=False)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2000, shuffle=True)
input_dims = 28 * 28
hidden_dims = 100
output_dims = 10
# train with sigmoid activation
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
model_sigmoid = Net(input_dims, hidden_dims, output_dims)
optimizer = torch.optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
training_result_sigmoid = train(model, criterion, train_loader, validation_loader, optimizer, epochs=10)
training_result_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=10)
optimizer = torch.optim.SGD(model_sigmoid.parameters(), lr=learning_rate)
optimizer
#[Out]# SGD (
#[Out]# Parameter Group 0
#[Out]#     dampening: 0
#[Out]#     differentiable: False
#[Out]#     foreach: None
#[Out]#     fused: None
#[Out]#     lr: 0.01
#[Out]#     maximize: False
#[Out]#     momentum: 0
#[Out]#     nesterov: False
#[Out]#     weight_decay: 0
#[Out]# )
training_result_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=10)
model_sigmoid.parameters()
#[Out]# <generator object Module.parameters at 0x00000294026CD380>
list(model_sigmoid.parameters())
#[Out]# [Parameter containing:
#[Out]#  tensor([[ 0.0194, -0.0266, -0.0279,  ...,  0.0060,  0.0113,  0.0085],
#[Out]#          [ 0.0123, -0.0336, -0.0181,  ..., -0.0157,  0.0083,  0.0211],
#[Out]#          [ 0.0189, -0.0318,  0.0302,  ..., -0.0212,  0.0233,  0.0312],
#[Out]#          ...,
#[Out]#          [ 0.0107, -0.0345,  0.0175,  ..., -0.0188, -0.0086,  0.0190],
#[Out]#          [ 0.0310,  0.0183, -0.0229,  ...,  0.0273,  0.0324, -0.0032],
#[Out]#          [-0.0010, -0.0243, -0.0024,  ..., -0.0182,  0.0152,  0.0204]],
#[Out]#         requires_grad=True),
#[Out]#  Parameter containing:
#[Out]#  tensor([-0.0231,  0.0063,  0.0321,  0.0354,  0.0100, -0.0071, -0.0037,  0.0196,
#[Out]#           0.0210,  0.0304, -0.0218,  0.0007, -0.0268,  0.0342,  0.0348,  0.0306,
#[Out]#          -0.0158,  0.0011, -0.0247, -0.0274,  0.0036, -0.0199,  0.0184,  0.0341,
#[Out]#           0.0251,  0.0011,  0.0135, -0.0251,  0.0131,  0.0326,  0.0219,  0.0177,
#[Out]#           0.0130, -0.0128, -0.0005, -0.0056,  0.0226,  0.0339, -0.0068, -0.0239,
#[Out]#           0.0010, -0.0036, -0.0283,  0.0082, -0.0331,  0.0017,  0.0324,  0.0300,
#[Out]#           0.0355, -0.0297, -0.0123, -0.0270,  0.0029,  0.0110, -0.0149,  0.0280,
#[Out]#          -0.0254,  0.0115,  0.0173, -0.0302,  0.0357, -0.0214, -0.0173, -0.0284,
#[Out]#          -0.0007,  0.0089, -0.0213, -0.0083, -0.0244,  0.0303, -0.0193,  0.0243,
#[Out]#          -0.0222,  0.0153,  0.0035,  0.0292,  0.0257,  0.0256,  0.0129, -0.0184,
#[Out]#          -0.0275, -0.0254,  0.0344, -0.0231, -0.0008,  0.0141, -0.0213,  0.0346,
#[Out]#           0.0092,  0.0078,  0.0258,  0.0067,  0.0033,  0.0342, -0.0086,  0.0242,
#[Out]#          -0.0029, -0.0109, -0.0244,  0.0297], requires_grad=True),
#[Out]#  Parameter containing:
#[Out]#  tensor([[ 9.0339e-03,  8.5571e-02, -6.1438e-02,  2.1236e-02,  3.7886e-02,
#[Out]#            2.9375e-02,  2.6985e-02, -7.5848e-02,  6.9926e-02, -9.2882e-02,
#[Out]#            8.9890e-02,  3.1338e-03,  5.6857e-02, -4.4686e-03,  7.2632e-02,
#[Out]#            1.2206e-02, -7.1508e-02, -6.7422e-02,  1.9868e-02,  1.3619e-02,
#[Out]#            6.6442e-02, -8.5472e-02, -8.6394e-02, -9.5965e-02,  2.8403e-02,
#[Out]#           -7.5900e-02, -5.0148e-02,  9.3684e-02,  4.9325e-02,  9.6515e-02,
#[Out]#            6.5703e-02,  7.7055e-02,  1.6195e-02,  6.4247e-03,  9.1580e-02,
#[Out]#           -3.3248e-02, -9.7063e-02,  9.2706e-02,  1.3161e-02, -4.3996e-02,
#[Out]#            4.5278e-02,  4.6913e-02, -4.7122e-02,  6.7754e-02, -8.8343e-02,
#[Out]#            4.2462e-02,  4.2758e-02, -1.3961e-02, -9.9975e-02,  1.5460e-02,
#[Out]#            5.6226e-02,  4.6425e-02,  4.4386e-02, -6.2824e-02,  1.3286e-02,
#[Out]#           -8.5086e-02, -4.0405e-02, -8.2322e-02,  7.0253e-02,  8.0958e-02,
#[Out]#            2.0304e-02,  1.4465e-02, -9.5264e-02, -1.1804e-02,  6.6776e-02,
#[Out]#           -1.8601e-03, -5.4295e-02,  5.8102e-02,  2.4665e-02, -2.7500e-02,
#[Out]#           -7.8517e-02, -9.7993e-02, -5.3755e-02,  5.4659e-02, -3.4978e-03,
#[Out]#            1.8482e-02, -4.0579e-02,  9.5457e-02,  1.2771e-02,  4.8219e-02,
#[Out]#           -2.9019e-02, -5.9180e-02,  5.0756e-02,  1.6411e-02, -2.9169e-02,
#[Out]#            6.5385e-02,  6.4370e-02,  7.9987e-02,  1.4050e-03, -9.9200e-02,
#[Out]#            3.2171e-03, -5.9692e-02,  5.2707e-02,  3.7460e-02, -3.6829e-02,
#[Out]#           -8.8708e-02,  2.1209e-02, -1.8791e-02,  4.4488e-02, -2.2527e-02],
#[Out]#          [-1.3171e-02,  3.8022e-02,  6.1957e-02, -3.8202e-03, -4.9740e-03,
#[Out]#           -2.0370e-02,  2.6676e-03,  6.3788e-02, -1.5068e-02, -1.7466e-02,
#[Out]#            9.0028e-02,  4.8243e-05,  6.9921e-02,  2.0676e-02, -3.2493e-02,
#[Out]#            1.5434e-02, -6.9547e-02, -6.1441e-02,  9.8231e-02, -8.0893e-02,
#[Out]#           -2.2774e-02,  8.5997e-02,  5.1195e-02, -8.0108e-02, -6.5858e-02,
#[Out]#           -5.8591e-02,  2.3217e-03, -1.2309e-02, -2.6099e-02,  3.6466e-02,
#[Out]#           -5.8812e-02, -8.8450e-02, -7.1272e-04,  4.1859e-03,  9.5194e-02,
#[Out]#           -6.5661e-02,  6.7447e-02,  4.0108e-02,  5.6376e-02,  1.7869e-02,
#[Out]#            8.9753e-02,  6.9059e-02, -1.7911e-02, -5.5415e-02,  1.2455e-02,
#[Out]#            6.8880e-02, -8.9468e-02, -6.6948e-02, -2.4655e-02, -6.7806e-02,
#[Out]#           -9.9359e-02, -2.6043e-02, -8.5800e-03, -5.4447e-02,  9.9423e-02,
#[Out]#           -3.7507e-02,  9.5282e-02, -5.4668e-03,  6.9856e-02, -2.7289e-02,
#[Out]#            8.9699e-02,  4.1479e-02, -4.5285e-02, -9.8969e-02,  3.3704e-02,
#[Out]#           -6.4080e-02, -4.6538e-02, -9.0846e-02,  5.7117e-02,  9.1080e-03,
#[Out]#            5.5047e-02,  1.0520e-02,  6.0214e-02,  6.6956e-02, -8.9181e-02,
#[Out]#           -9.9993e-02, -4.5854e-02,  2.2014e-02,  8.1952e-02, -5.3317e-02,
#[Out]#            2.2961e-02, -9.5869e-02, -9.4763e-02, -8.6528e-02, -9.9207e-03,
#[Out]#           -7.6134e-02, -4.6599e-02,  6.0258e-03,  3.9655e-03,  5.0602e-02,
#[Out]#           -5.4293e-02, -8.0310e-02, -4.9499e-02, -6.9885e-02,  6.6944e-02,
#[Out]#           -8.6071e-02, -7.7599e-02,  5.1676e-02, -3.7488e-02, -5.7003e-03],
#[Out]#          [-5.5278e-02,  7.3915e-02, -6.5363e-03, -1.9293e-02, -4.2480e-02,
#[Out]#           -5.6088e-02, -2.7621e-03, -8.0145e-03,  4.2960e-02, -5.1870e-02,
#[Out]#           -1.1579e-02, -4.8059e-02,  6.6330e-02, -9.3992e-02, -2.6974e-02,
#[Out]#           -1.5188e-02,  5.8434e-02, -5.6767e-02, -6.7619e-02, -9.4166e-02,
#[Out]#           -5.4554e-03, -6.7896e-02,  5.0275e-02, -6.0964e-03, -7.5809e-02,
#[Out]#            1.1078e-02,  1.1129e-02,  2.3779e-02, -5.7827e-02, -9.4045e-02,
#[Out]#           -6.7321e-02,  3.9556e-02,  1.6507e-03,  3.9419e-02, -4.6716e-02,
#[Out]#            2.3483e-02,  5.3473e-02,  2.1920e-02, -2.2834e-02,  6.4785e-02,
#[Out]#            9.7301e-02, -1.5280e-02,  7.7451e-02,  3.5877e-02,  6.1293e-02,
#[Out]#           -2.2573e-02, -6.3378e-02, -5.6372e-02,  3.3313e-03,  4.7280e-02,
#[Out]#           -8.5005e-02,  6.6074e-02, -2.4968e-02, -6.5087e-02, -5.9585e-04,
#[Out]#            7.2717e-02,  6.6330e-02,  1.4822e-02,  4.3909e-02, -6.7262e-02,
#[Out]#            6.8808e-02,  4.9253e-02,  2.1893e-02,  3.4714e-02, -8.1322e-02,
#[Out]#           -5.6766e-02, -6.2915e-02,  9.9964e-02, -5.3793e-02, -7.4696e-02,
#[Out]#           -9.1621e-02, -9.6285e-02,  8.4958e-02, -4.2940e-02, -4.4439e-02,
#[Out]#            6.5168e-02, -3.1706e-02, -4.8176e-02,  3.7991e-02,  2.4550e-02,
#[Out]#            7.5054e-02, -9.4366e-02, -1.5407e-02, -8.5994e-02,  1.4057e-02,
#[Out]#            2.2050e-02, -3.9039e-02,  2.4454e-02, -4.8936e-02,  7.6242e-02,
#[Out]#            3.9913e-03,  5.8861e-02, -2.3273e-02,  9.8418e-03, -2.2214e-02,
#[Out]#            3.2548e-02, -9.8889e-02, -9.2337e-02, -7.5733e-02,  9.6672e-02],
#[Out]#          [ 4.2119e-02, -7.7380e-03, -2.0049e-02, -4.7739e-02, -6.6263e-02,
#[Out]#           -7.8660e-02,  2.0821e-02,  1.3372e-02, -9.5197e-02,  3.3326e-02,
#[Out]#            6.0459e-02, -9.2363e-02, -3.5140e-02, -7.6275e-03,  6.4264e-02,
#[Out]#            2.4827e-02,  3.7425e-03, -6.4084e-02, -6.0149e-02,  5.6060e-02,
#[Out]#            2.2099e-03, -9.7887e-02, -8.3320e-03, -9.9078e-02, -1.1172e-02,
#[Out]#           -1.8517e-02, -1.1460e-02,  1.1451e-02,  5.1946e-02,  3.6707e-02,
#[Out]#           -4.2599e-02, -1.0380e-02, -3.0561e-02,  5.3630e-02,  4.0932e-02,
#[Out]#           -3.3388e-02,  7.2661e-03,  7.7136e-02,  5.9734e-02,  6.6915e-02,
#[Out]#           -9.3776e-02, -3.8051e-02, -8.7589e-02,  7.1014e-02, -2.5893e-02,
#[Out]#            8.6321e-02, -3.9661e-02, -1.3761e-02,  5.2162e-02,  4.9787e-02,
#[Out]#           -3.6241e-02, -6.2748e-02,  8.9333e-02, -8.4711e-02,  5.5211e-02,
#[Out]#            3.3305e-02, -8.1102e-02, -4.8443e-02,  5.4236e-02,  5.4550e-02,
#[Out]#            8.6002e-02,  9.5574e-02, -3.3585e-02, -1.0194e-02,  8.1805e-02,
#[Out]#            3.6810e-02, -8.8583e-02,  2.3169e-02, -7.3960e-02,  8.1881e-02,
#[Out]#           -5.1386e-02,  1.5575e-02, -7.7453e-02,  8.7360e-02,  2.9392e-02,
#[Out]#            3.2921e-02, -5.0205e-02,  2.4769e-02,  8.6323e-02, -8.3282e-02,
#[Out]#           -3.0176e-02,  4.7557e-02,  7.6762e-02, -2.4785e-02, -4.6393e-02,
#[Out]#            1.4724e-02, -7.6621e-02,  9.9307e-02,  8.7722e-02,  6.1609e-02,
#[Out]#           -3.8403e-02, -7.4371e-02, -3.7471e-02, -7.1941e-02,  9.0624e-02,
#[Out]#            6.6742e-02, -6.1282e-02, -3.3752e-03,  9.5172e-02, -4.0474e-02],
#[Out]#          [ 7.3334e-02, -5.3652e-02,  6.8253e-02, -1.2646e-02,  9.7712e-03,
#[Out]#            2.9845e-02, -7.1784e-02, -3.5989e-02, -5.5368e-02,  9.0325e-02,
#[Out]#           -5.9724e-02,  2.1618e-03,  2.8053e-02,  5.9388e-02,  7.1163e-02,
#[Out]#            4.7757e-02,  5.5934e-02, -1.5649e-02, -5.8303e-02, -2.8163e-02,
#[Out]#            7.2348e-02, -4.2009e-02,  5.4384e-02,  7.2092e-02, -3.1489e-02,
#[Out]#            5.1882e-02, -7.7873e-02, -5.0829e-02, -2.5621e-02,  2.0756e-04,
#[Out]#           -7.4801e-02,  5.1905e-02,  9.9513e-02,  2.9156e-02, -7.5184e-02,
#[Out]#           -9.7335e-02, -8.7758e-02, -8.8487e-02,  5.5338e-02,  8.8676e-02,
#[Out]#           -2.1545e-02, -8.1087e-02,  8.0355e-02, -2.1860e-02, -3.4277e-02,
#[Out]#            5.8072e-02, -1.8491e-02, -4.1984e-02,  5.5665e-02, -1.5786e-02,
#[Out]#            5.2955e-02, -8.2165e-02, -5.8585e-02, -4.8301e-02, -3.8848e-02,
#[Out]#            8.0996e-03, -1.2501e-02,  7.4193e-02,  5.8986e-02,  1.4943e-02,
#[Out]#           -6.6688e-02, -3.3078e-02,  4.0368e-03, -6.1886e-02, -5.7237e-02,
#[Out]#           -5.9864e-03, -1.4440e-02, -1.0311e-02,  2.8350e-02,  4.5026e-02,
#[Out]#            7.1339e-02, -9.0337e-02, -1.9969e-02, -6.1156e-02,  1.9198e-02,
#[Out]#            2.7865e-03, -8.5070e-02, -6.6885e-02, -3.2358e-02,  2.2159e-02,
#[Out]#           -3.8960e-02,  6.0257e-02,  5.2562e-02, -9.7111e-05,  9.1818e-02,
#[Out]#            7.9194e-02,  9.2542e-02,  1.7472e-02, -9.9366e-02,  8.1847e-02,
#[Out]#           -4.1269e-02,  1.5830e-02,  5.9800e-02,  5.9295e-02,  8.3852e-02,
#[Out]#           -9.6795e-02,  4.0739e-02, -5.2674e-02,  1.0887e-02, -8.6024e-02],
#[Out]#          [-2.9310e-02,  9.0414e-02, -7.4825e-02, -3.0848e-02,  5.0684e-02,
#[Out]#            3.9313e-02, -3.6428e-02,  2.0455e-02, -6.8068e-02, -7.3165e-02,
#[Out]#           -2.0853e-02,  7.8844e-02,  9.3677e-02, -6.1343e-03,  8.2537e-02,
#[Out]#           -5.2696e-02,  4.3909e-02,  7.3990e-02,  3.9760e-02,  5.7528e-02,
#[Out]#           -4.0150e-02,  1.0710e-02, -1.2479e-02,  5.9231e-02,  2.5129e-02,
#[Out]#           -6.6200e-02,  6.5640e-02,  7.9854e-02,  6.5179e-02,  1.5992e-02,
#[Out]#            7.5698e-02, -2.3189e-02, -8.0350e-03,  4.5241e-02, -3.5489e-02,
#[Out]#            3.9151e-02, -3.6277e-02, -1.2395e-02, -2.4248e-02, -8.9542e-02,
#[Out]#            1.5720e-02,  5.5277e-02,  5.5093e-02,  7.5667e-02,  2.0164e-02,
#[Out]#           -4.9490e-02,  1.4875e-02, -8.8954e-02,  2.3586e-02, -8.7448e-02,
#[Out]#           -2.9854e-02,  5.4885e-03,  3.3299e-03, -8.7103e-02,  5.9005e-02,
#[Out]#            6.4070e-02,  9.2317e-02, -1.4882e-03, -8.9190e-02,  6.8305e-02,
#[Out]#            6.3983e-03,  2.0500e-02,  9.6909e-02,  3.0410e-02, -6.7538e-02,
#[Out]#            1.0762e-02, -8.8766e-02, -9.1558e-02,  5.3177e-02,  5.8385e-02,
#[Out]#           -2.9932e-02, -7.8984e-02, -2.4322e-02,  4.2846e-02, -1.6901e-02,
#[Out]#            3.5574e-02, -1.0396e-02,  7.5227e-02,  6.2357e-02, -4.9341e-02,
#[Out]#            4.1680e-02,  7.0494e-02,  4.5892e-03,  9.5231e-02,  7.2434e-02,
#[Out]#           -6.2312e-02,  3.2192e-02, -4.2161e-02, -7.9222e-02,  3.8558e-02,
#[Out]#           -3.6097e-03, -9.1805e-02, -3.6727e-02, -8.0572e-02,  4.6624e-02,
#[Out]#           -2.2572e-02,  1.3918e-02, -8.7570e-02, -3.8376e-02,  6.1511e-02],
#[Out]#          [ 3.0466e-02, -5.7229e-02, -3.8022e-02, -4.3047e-02, -5.3816e-02,
#[Out]#            6.1133e-02, -8.9032e-02, -6.9141e-02,  8.3682e-02, -4.4476e-02,
#[Out]#           -1.8952e-02,  2.1824e-02,  7.9979e-02, -2.7083e-02, -4.6823e-02,
#[Out]#            7.7351e-02, -9.3804e-02,  9.9644e-02,  5.9867e-02, -2.1365e-02,
#[Out]#           -9.9114e-02,  6.1670e-02, -6.9547e-02,  5.8943e-02, -9.1088e-02,
#[Out]#            3.1972e-02, -2.0633e-02,  1.6153e-02,  2.6799e-02,  6.8467e-02,
#[Out]#            9.0686e-03,  1.8445e-02,  9.1378e-03,  3.7800e-03,  3.0789e-02,
#[Out]#            4.4666e-02, -2.4167e-02, -5.5649e-02, -4.3872e-02,  6.3559e-02,
#[Out]#            8.6460e-02,  8.8836e-02, -2.0459e-02,  8.6780e-02,  8.5191e-02,
#[Out]#            5.8645e-02, -1.0742e-03, -5.9350e-02,  2.6208e-02,  1.3456e-02,
#[Out]#           -8.5010e-02, -2.5571e-02, -8.3638e-02, -4.2061e-02,  4.1604e-02,
#[Out]#           -3.5212e-02, -2.2004e-02, -8.3920e-02, -4.5386e-02, -4.7726e-02,
#[Out]#           -4.5334e-02, -1.3476e-02,  8.4544e-02, -9.0059e-03,  5.5923e-02,
#[Out]#            5.2113e-02, -6.0648e-02,  8.3295e-02, -3.3774e-02, -9.6796e-02,
#[Out]#           -9.2565e-02, -3.3339e-02, -1.7130e-02, -1.5548e-02,  3.6692e-02,
#[Out]#            6.8657e-02,  9.9075e-02, -4.1886e-02, -7.0964e-02,  9.1063e-02,
#[Out]#            5.6157e-02,  7.0478e-02,  8.7652e-02,  1.6280e-02,  8.5966e-02,
#[Out]#            7.5712e-02, -9.8314e-02,  1.8165e-02, -7.7405e-02,  7.2978e-02,
#[Out]#           -6.9057e-02, -6.2248e-02, -6.7754e-02,  3.0474e-02,  9.8025e-02,
#[Out]#            8.8781e-02,  3.5396e-02,  2.8571e-02, -7.9856e-02, -1.2305e-02],
#[Out]#          [-3.7735e-02, -2.8649e-03,  9.8877e-02,  1.4040e-02,  6.6646e-02,
#[Out]#           -8.7187e-02, -3.9292e-02, -1.3315e-02, -3.8013e-03,  1.8612e-02,
#[Out]#            5.3640e-02,  7.3629e-02, -2.9512e-04, -9.3342e-02, -4.4688e-02,
#[Out]#           -1.1206e-02, -5.7092e-02,  7.6158e-03,  3.1979e-02,  2.7258e-02,
#[Out]#            6.5850e-02,  4.2882e-02,  1.1215e-02, -6.7112e-02,  2.4093e-02,
#[Out]#           -5.1018e-02,  2.5588e-02, -9.7480e-03, -6.0646e-02, -9.7127e-02,
#[Out]#            6.4295e-02, -5.1583e-02,  8.9654e-02,  8.5318e-02, -6.0133e-02,
#[Out]#            9.0845e-03, -7.5681e-02,  6.4462e-02,  1.2026e-02, -1.6473e-02,
#[Out]#           -7.9326e-02,  9.3638e-02,  7.5757e-02,  6.1620e-02,  7.0839e-03,
#[Out]#            2.0953e-02,  2.1012e-02,  1.7944e-02,  2.1466e-02, -2.4651e-02,
#[Out]#           -2.0993e-02, -6.8421e-02, -2.4408e-02, -9.7580e-02, -3.9588e-03,
#[Out]#           -9.2681e-02, -9.9968e-02, -9.8541e-02, -9.4755e-02, -8.0108e-02,
#[Out]#           -8.8065e-02,  6.1488e-02,  4.6192e-02, -2.5181e-02, -1.7564e-02,
#[Out]#            7.5220e-02, -5.1301e-02,  7.3828e-02,  2.6225e-02,  6.1284e-02,
#[Out]#            2.6820e-03, -9.3789e-03,  4.5974e-02, -4.9498e-02, -2.0485e-02,
#[Out]#           -7.5619e-02,  5.3833e-02, -9.2501e-03, -2.3061e-02, -8.8510e-02,
#[Out]#            3.3938e-02,  4.4238e-03, -6.8918e-06, -5.2000e-02, -3.4271e-02,
#[Out]#            8.3079e-02,  3.1139e-03,  2.5093e-02, -6.0842e-02, -3.1945e-02,
#[Out]#            2.4755e-02, -1.7304e-02,  9.8173e-02,  5.1080e-02,  5.0767e-02,
#[Out]#           -6.0529e-02, -6.3363e-03,  7.1006e-02, -3.1165e-02, -3.7294e-02],
#[Out]#          [-3.2859e-02,  5.3377e-02, -6.5881e-02,  6.9167e-02, -7.9011e-02,
#[Out]#           -1.8116e-02,  5.2008e-02,  2.6768e-02,  2.5508e-02, -1.8742e-02,
#[Out]#            2.2377e-02, -3.6013e-02, -2.4532e-02,  2.6747e-02, -6.2504e-02,
#[Out]#           -4.9625e-02,  3.6196e-02,  4.3433e-02,  1.0800e-02,  3.0482e-02,
#[Out]#           -6.7122e-02,  7.7147e-02,  3.8550e-02, -7.4884e-02, -9.1548e-02,
#[Out]#           -8.7637e-02,  4.3459e-02, -7.7646e-02, -5.4561e-02,  4.7784e-02,
#[Out]#            1.0970e-02,  4.7704e-03,  4.8169e-02,  1.5179e-02,  6.6578e-02,
#[Out]#            4.6459e-02,  9.2580e-02,  3.7205e-02,  3.1503e-02,  5.4531e-02,
#[Out]#            6.1807e-02, -1.5221e-02, -1.5391e-02,  2.1689e-02, -7.3633e-02,
#[Out]#           -3.7299e-02,  3.6033e-02,  4.2239e-03,  2.0803e-02,  3.0808e-02,
#[Out]#            5.5557e-02, -8.6445e-02, -4.0507e-03, -9.6681e-02,  2.8832e-02,
#[Out]#           -5.5386e-02,  8.6986e-02,  6.1892e-03, -3.5848e-02, -2.4481e-02,
#[Out]#            4.4304e-02, -5.7463e-02, -3.4299e-03, -5.2951e-02, -3.6987e-02,
#[Out]#           -7.0352e-02,  4.3512e-02, -9.3008e-02,  7.1586e-02, -4.0563e-02,
#[Out]#            8.8351e-02,  6.6255e-02, -7.8014e-02,  6.9256e-02,  5.0761e-02,
#[Out]#            7.2907e-02,  3.9502e-02,  9.2542e-02,  5.9944e-02,  2.3909e-02,
#[Out]#           -2.9570e-02, -3.2663e-02,  8.1314e-02, -9.9223e-02,  9.2864e-02,
#[Out]#            3.0134e-02, -6.8250e-02,  3.8046e-02, -5.7602e-02, -6.1137e-02,
#[Out]#            8.0316e-02, -7.9420e-02, -6.6410e-02, -4.1546e-04, -9.9685e-02,
#[Out]#            3.2016e-02,  6.1602e-02, -5.5685e-02, -5.0674e-02, -1.8161e-02],
#[Out]#          [ 5.6220e-02, -5.6539e-02, -3.2765e-02, -9.2100e-02,  6.5388e-02,
#[Out]#            4.1940e-02,  7.8088e-02,  6.2488e-02, -4.5015e-02, -5.3051e-02,
#[Out]#            5.4821e-02, -2.3003e-02, -4.4623e-02, -9.2062e-02,  6.4691e-02,
#[Out]#            8.4719e-02,  4.8444e-02, -4.1096e-02, -1.8045e-02, -6.0001e-02,
#[Out]#           -3.5226e-02, -2.7237e-02,  3.4885e-02, -8.0660e-02, -9.5977e-02,
#[Out]#           -6.1100e-02, -5.3462e-02, -6.6748e-02,  4.0244e-02,  5.6986e-02,
#[Out]#           -2.3266e-02,  9.5356e-02,  9.7377e-02,  5.9798e-02, -1.7202e-03,
#[Out]#           -2.6441e-02,  5.2759e-02,  4.9174e-02,  2.0941e-02,  2.6360e-02,
#[Out]#            6.1897e-02, -7.9438e-02, -9.7894e-02, -9.3477e-02,  1.8520e-02,
#[Out]#           -3.1903e-02,  5.7594e-03,  5.2131e-03,  2.0419e-02,  3.3874e-02,
#[Out]#           -7.7628e-02,  7.9674e-02,  6.3023e-02, -8.6089e-02,  6.7635e-02,
#[Out]#            8.9191e-02,  5.2941e-02,  6.8904e-02, -5.6268e-03,  1.8785e-02,
#[Out]#            1.7106e-02, -4.9424e-02, -7.1414e-03, -4.2312e-02,  2.5456e-02,
#[Out]#           -7.5392e-02, -5.4746e-02, -4.5042e-02,  7.6577e-02, -9.4824e-02,
#[Out]#            4.5185e-02,  3.8210e-02, -8.4850e-02,  1.2807e-02, -6.3644e-02,
#[Out]#            2.8604e-02, -3.0182e-02,  8.5445e-03,  9.4760e-02,  4.4048e-02,
#[Out]#           -1.2181e-03, -7.2370e-02,  5.2113e-02,  4.3908e-02,  3.2071e-02,
#[Out]#            7.2845e-02, -1.8985e-03, -6.7522e-02,  3.0820e-02,  1.7142e-03,
#[Out]#            6.2889e-02,  3.1454e-02,  6.7862e-02, -4.4096e-02,  3.4086e-02,
#[Out]#            8.1698e-02, -8.7306e-03, -6.2536e-02,  2.0487e-02,  4.4322e-02]],
#[Out]#         requires_grad=True),
#[Out]#  Parameter containing:
#[Out]#  tensor([-0.0864,  0.0676, -0.0872, -0.0908,  0.0414,  0.0530, -0.0561, -0.0528,
#[Out]#           0.0352,  0.0671], requires_grad=True)]
optimizer
#[Out]# SGD (
#[Out]# Parameter Group 0
#[Out]#     dampening: 0
#[Out]#     differentiable: False
#[Out]#     foreach: None
#[Out]#     fused: None
#[Out]#     lr: 0.01
#[Out]#     maximize: False
#[Out]#     momentum: 0
#[Out]#     nesterov: False
#[Out]#     weight_decay: 0
#[Out]# )
def train(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    useful_stuff = {
    'training_loss': [],
    'validation_accuracy': [],
    }
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            useful_stuff['training_loss'].append(loss.item())
        correct = 0
        for x, y in validation_loader:
            z = model(x.view(-1,28*28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = (correct / len(validation_dataset)) * 100
        useful_stuff['validation_accuracy'].append(accuracy)
    return useful_stuff
    
training_result_sigmoid = train(model_sigmoid, criterion, train_loader, validation_loader, optimizer, epochs=10)
model_tanh = NetTanh(input_dims, hidden_dims, output_dims)
optimizer = torch.optim.SGD(model.tanh.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model_tanh.parameters(), lr=learning_rate)
training_result_tanh = train(model_tanh, criterion, train_loader, validation_loader, optimizer, epochs=10)
model_relu = NetRelu(input_dims, hidden_dims, output_dims)
optimizer = torch.optim.SGD(model_reul.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model_relu.parameters(), lr=learning_rate)
training_result_relu = train(model_relu, criterion, train_loader, validation_loader, optimizer, epochs=10)
plt.plot(training_result_sigmoid['training_loss'], label='sigmoid')
#[Out]# [<matplotlib.lines.Line2D at 0x29402859340>]
plt.plot(training_result_tanh['training_loss'], label='tanh')
#[Out]# [<matplotlib.lines.Line2D at 0x294609badb0>]
plt.plot(training_result_relu['training_loss'], label='relu')
#[Out]# [<matplotlib.lines.Line2D at 0x294028fa090>]
plt.ylabel('loss')
#[Out]# Text(0, 0.5, 'loss')
plt.title('training loss iterarion')
#[Out]# Text(0.5, 1.0, 'training loss iterarion')
plt.legend()
#[Out]# <matplotlib.legend.Legend at 0x294028fb350>
plt.show()
plt.plot(training_results_tanch['validation_accuracy'], label='tanh')
plt.plot(training_results['validation_accuracy'], label='sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label='relu') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')
plt.legend()
plt.show()
plt.plot(training_result_tanh['validation_accuracy'], label='tanh')
plt.plot(training_result_sigmoid['validation_accuracy'], label='sigmoid')
plt.plot(training_results_relu['validation_accuracy'], label='relu') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')
plt.legend()
plt.show()
plt.plot(training_result_tanh['validation_accuracy'], label='tanh')
plt.plot(training_result_sigmoid['validation_accuracy'], label='sigmoid')
plt.plot(training_result_relu['validation_accuracy'], label='relu') 
plt.ylabel('validation accuracy')
plt.xlabel('epochs ')
plt.legend()
plt.show()
get_ipython().run_line_magic('logstop', '')
