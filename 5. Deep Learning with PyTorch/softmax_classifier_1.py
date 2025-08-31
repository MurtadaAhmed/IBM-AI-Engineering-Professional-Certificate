# IPython log file

import torch.nn as nn
import torch
import matplotlib.pyplot as plt 
import numpy as np
from torch.utils.data import Dataset, DataLoader
def plot_data(data_set, model = None, n = 1, color = False):
    X = data_set[:][0]
    Y = data_set[:][1]
    plt.plot(X[Y == 0, 0].numpy(), Y[Y == 0].numpy(), 'bo', label = 'y = 0')
    plt.plot(X[Y == 1, 0].numpy(), 0 * Y[Y == 1].numpy(), 'ro', label = 'y = 1')
    plt.plot(X[Y == 2, 0].numpy(), 0 * Y[Y == 2].numpy(), 'go', label = 'y = 2')
    plt.ylim((-0.1, 3))
    plt.legend()
    if model != None:
        w = list(model.parameters())[0][0].detach()
        b = list(model.parameters())[1][0].detach()
        y_label = ['yhat=0', 'yhat=1', 'yhat=2']
        y_color = ['b', 'r', 'g']
        Y = []
        for w, b, y_l, y_c in zip(model.state_dict()['0.weight'], model.state_dict()['0.bias'], y_label, y_color):
            Y.append((w * X + b).numpy())
            plt.plot(X.numpy(), (w * X + b).numpy(), y_c, label = y_l)
        if color == True:
            x = X.numpy()
            x = x.reshape(-1)
            top = np.ones(x.shape)
            y0 = Y[0].reshape(-1)
            y1 = Y[1].reshape(-1)
            y2 = Y[2].reshape(-1)
            plt.fill_between(x, y0, where = y1 > y1, interpolate = True, color = 'blue')
            plt.fill_between(x, y0, where = y1 > y2, interpolate = True, color = 'blue')
            plt.fill_between(x, y1, where = y1 > y0, interpolate = True, color = 'red')
            plt.fill_between(x, y1, where = ((y1 > y2) * (y1 > y0)),interpolate = True, color = 'red')
            plt.fill_between(x, y2, where = (y2 > y0) * (y0 > 0),interpolate = True, color = 'green')
            plt.fill_between(x, y2, where = (y2 > y1), interpolate = True, color = 'green')
    plt.legend()
    plt.show()
    
torch.manual_seed(0)
#[Out]# <torch._C.Generator at 0x220fed07fd0>
class Data(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2
        self.y = self.y.type(torch.LongTensor)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len
        
data_set = Data()
data_set.x
plot_data(data_set)
model = nn.Sequential(nn.Linear(1, 3))
model.state_dict()
#[Out]# OrderedDict([('0.weight',
#[Out]#               tensor([[-0.0075],
#[Out]#                       [ 0.5364],
#[Out]#                       [-0.8230]])),
#[Out]#              ('0.bias', tensor([-0.7359, -0.3852,  0.2682]))])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
trainloader = DataLoader(dataset = data_set, batch_size = 5)
LOSS = []
def train_model(epochs):
    for epoch in range(epochs):
        if epoch % 50 == 0:
            pass
            plot_data(data_set, model)
        for x, y in trainloader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss)
            loss.backward()
            optimizer.step()
train_model(300)
z =  model(data_set.x)
_, yhat = z.max(1)
print("The prediction:", yhat)
z
#[Out]# tensor([[ 3.7019,  1.1621, -5.1287],
#[Out]#         [ 3.4652,  1.1572, -4.9166],
#[Out]#         [ 3.2285,  1.1523, -4.7044],
#[Out]#         [ 2.9918,  1.1474, -4.4922],
#[Out]#         [ 2.7551,  1.1425, -4.2800],
#[Out]#         [ 2.5184,  1.1376, -4.0678],
#[Out]#         [ 2.2818,  1.1326, -3.8556],
#[Out]#         [ 2.0451,  1.1277, -3.6434],
#[Out]#         [ 1.8084,  1.1228, -3.4312],
#[Out]#         [ 1.5717,  1.1179, -3.2190],
#[Out]#         [ 1.3350,  1.1130, -3.0069],
#[Out]#         [ 1.0983,  1.1081, -2.7947],
#[Out]#         [ 0.8617,  1.1032, -2.5825],
#[Out]#         [ 0.6250,  1.0982, -2.3703],
#[Out]#         [ 0.3883,  1.0933, -2.1581],
#[Out]#         [ 0.1516,  1.0884, -1.9459],
#[Out]#         [-0.0851,  1.0835, -1.7337],
#[Out]#         [-0.3218,  1.0786, -1.5215],
#[Out]#         [-0.5584,  1.0737, -1.3093],
#[Out]#         [-0.7951,  1.0688, -1.0971],
#[Out]#         [-1.0318,  1.0638, -0.8850],
#[Out]#         [-1.2685,  1.0589, -0.6728],
#[Out]#         [-1.5052,  1.0540, -0.4606],
#[Out]#         [-1.7419,  1.0491, -0.2484],
#[Out]#         [-1.9786,  1.0442, -0.0362],
#[Out]#         [-2.2152,  1.0393,  0.1760],
#[Out]#         [-2.4519,  1.0344,  0.3882],
#[Out]#         [-2.6886,  1.0294,  0.6004],
#[Out]#         [-2.9253,  1.0245,  0.8126],
#[Out]#         [-3.1620,  1.0196,  1.0247],
#[Out]#         [-3.3987,  1.0147,  1.2369],
#[Out]#         [-3.6353,  1.0098,  1.4491],
#[Out]#         [-3.8720,  1.0049,  1.6613],
#[Out]#         [-4.1087,  1.0000,  1.8735],
#[Out]#         [-4.3454,  0.9950,  2.0857],
#[Out]#         [-4.5821,  0.9901,  2.2979],
#[Out]#         [-4.8188,  0.9852,  2.5101],
#[Out]#         [-5.0554,  0.9803,  2.7223],
#[Out]#         [-5.2921,  0.9754,  2.9344],
#[Out]#         [-5.5288,  0.9705,  3.1466]], grad_fn=<AddmmBackward0>)
z.max()
#[Out]# tensor(3.7019, grad_fn=<MaxBackward1>)
z.max(1)
#[Out]# torch.return_types.max(
#[Out]# values=tensor([3.7019, 3.4652, 3.2285, 2.9918, 2.7551, 2.5184, 2.2818, 2.0451, 1.8084,
#[Out]#         1.5717, 1.3350, 1.1081, 1.1032, 1.0982, 1.0933, 1.0884, 1.0835, 1.0786,
#[Out]#         1.0737, 1.0688, 1.0638, 1.0589, 1.0540, 1.0491, 1.0442, 1.0393, 1.0344,
#[Out]#         1.0294, 1.0245, 1.0247, 1.2369, 1.4491, 1.6613, 1.8735, 2.0857, 2.2979,
#[Out]#         2.5101, 2.7223, 2.9344, 3.1466], grad_fn=<MaxBackward0>),
#[Out]# indices=tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#[Out]#         1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]))
correct = (data_set.y == yhat).sum().item()
accuracy = correct / len(data_set)
print("The accuracy: ", accuracy)
Softmax_fn=nn.Softmax(dim=-1)
Probability =Softmax_fn(z)
Probability
#[Out]# tensor([[9.2675e-01, 7.3110e-02, 1.3548e-04],
#[Out]#         [9.0934e-01, 9.0447e-02, 2.0825e-04],
#[Out]#         [8.8828e-01, 1.1140e-01, 3.1868e-04],
#[Out]#         [8.6305e-01, 1.3646e-01, 4.8505e-04],
#[Out]#         [8.3317e-01, 1.6610e-01, 7.3354e-04],
#[Out]#         [7.9825e-01, 2.0065e-01, 1.1010e-03],
#[Out]#         [7.5810e-01, 2.4026e-01, 1.6380e-03],
#[Out]#         [7.1278e-01, 2.8481e-01, 2.4125e-03],
#[Out]#         [6.6264e-01, 3.3384e-01, 3.5135e-03],
#[Out]#         [6.0845e-01, 3.8649e-01, 5.0540e-03],
#[Out]#         [5.5130e-01, 4.4153e-01, 7.1736e-03],
#[Out]#         [4.9257e-01, 4.9739e-01, 1.0041e-02],
#[Out]#         [4.3382e-01, 5.5233e-01, 1.3853e-02],
#[Out]#         [3.7661e-01, 6.0455e-01, 1.8840e-02],
#[Out]#         [3.2234e-01, 6.5240e-01, 2.5260e-02],
#[Out]#         [2.7214e-01, 6.9445e-01, 3.3408e-02],
#[Out]#         [2.2677e-01, 7.2962e-01, 4.3611e-02],
#[Out]#         [1.8664e-01, 7.5713e-01, 5.6228e-02],
#[Out]#         [1.5182e-01, 7.7653e-01, 7.1652e-02],
#[Out]#         [1.2213e-01, 7.8758e-01, 9.0293e-02],
#[Out]#         [9.7191e-02, 7.9024e-01, 1.1257e-01],
#[Out]#         [7.6535e-02, 7.8460e-01, 1.3886e-01],
#[Out]#         [5.9638e-02, 7.7085e-01, 1.6951e-01],
#[Out]#         [4.5978e-02, 7.4930e-01, 2.0472e-01],
#[Out]#         [3.5060e-02, 7.2039e-01, 2.4455e-01],
#[Out]#         [2.6431e-02, 6.8476e-01, 2.8881e-01],
#[Out]#         [1.9692e-02, 6.4323e-01, 3.3708e-01],
#[Out]#         [1.4493e-02, 5.9688e-01, 3.8863e-01],
#[Out]#         [1.0534e-02, 5.4697e-01, 4.4249e-01],
#[Out]#         [7.5600e-03, 4.9495e-01, 4.9749e-01],
#[Out]#         [5.3581e-03, 4.4229e-01, 5.5236e-01],
#[Out]#         [3.7514e-03, 3.9043e-01, 6.0582e-01],
#[Out]#         [2.5960e-03, 3.4065e-01, 6.5675e-01],
#[Out]#         [1.7769e-03, 2.9399e-01, 7.0423e-01],
#[Out]#         [1.2042e-03, 2.5119e-01, 7.4760e-01],
#[Out]#         [8.0867e-04, 2.1269e-01, 7.8650e-01],
#[Out]#         [5.3873e-04, 1.7865e-01, 8.2081e-01],
#[Out]#         [3.5639e-04, 1.4901e-01, 8.5063e-01],
#[Out]#         [2.3435e-04, 1.2354e-01, 8.7623e-01],
#[Out]#         [1.5330e-04, 1.0190e-01, 8.9795e-01]], grad_fn=<SoftmaxBackward0>)
for i in range(3):
    print("probability of class {} isg given by  {}".format(i, Probability[0,i]) )
    
get_ipython().run_line_magic('logstop', '')
