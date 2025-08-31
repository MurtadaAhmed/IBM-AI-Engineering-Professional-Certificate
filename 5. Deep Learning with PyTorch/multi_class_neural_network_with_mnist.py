# IPython log file

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
c = get_config()
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')
ipython profile create
get_ipython().run_line_magic('logstop', '')
