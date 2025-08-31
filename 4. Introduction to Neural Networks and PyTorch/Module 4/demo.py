"""
Dataset
Dataloader

nn.Linear(input_features, output_features)
nn.MSELoss
optim.SGD

optimizer.zero_grad()
loss.backward()
optimizer.step()
---------------------
Multiple linear regression
- bias
- w1, w2, wn

x is like a row of features
w is like a column of parameters
----
we so dot product operation between x and w (columns in x equals to the rows of w)
"""
from torch.nn  import Linear
import torch
torch.manual_seed(1)
model = Linear(in_features=2, out_features=1)
list(model.parameters()) # gives us the w and the b
model.state_dict()
X = torch.tensor([1.0, 3.0])
yhat = model(X)
