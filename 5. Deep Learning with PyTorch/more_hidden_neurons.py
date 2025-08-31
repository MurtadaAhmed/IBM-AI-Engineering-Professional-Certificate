# IPython log file

get_ipython().run_line_magic('cls', '')
import torch
torch.linspace(-3, 3, 1)
#[Out]# tensor([-3.])
torch.linspace(-3, 3, 5)
#[Out]# tensor([-3.0000, -1.5000,  0.0000,  1.5000,  3.0000])
torch.arange(-3, 3, 5)
#[Out]# tensor([-3,  2])
get_ipython().run_line_magic('logstop', '')
