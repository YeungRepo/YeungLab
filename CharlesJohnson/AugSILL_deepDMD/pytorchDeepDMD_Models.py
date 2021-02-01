import time
import numpy as np
from torch import nn
import torch#._C as torch 

class Feedforward(nn.Module):
    """
    This class is a feedforward neural network. It has a vector input of length dim_in, 
    and the output is of length dim_in + dim_added.
    """
    def __init__(self, n_layers, layer_width, dim_in, dim_added, activation=nn.ReLU6):
        nn.Module.__init__(self)
        self.activation = activation()
        layers = []
        layers.append(nn.Linear(dim_in, layer_width))
        for _i in range(n_layers):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(self.activation)
        layers.append(nn.Linear(layer_width, dim_in + dim_added))
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class feedforward_DDMD_no_input(nn.Module):
    """
    This class is a feedforward neural network that output the unbiased state-inclusive dictionary, then a 
    linear operator is applied (note bias=False) to get the lifting one step in the future.
    """
    def __init__(self, n_layers, layer_width, dim_y, dim_added_y):
        nn.Module.__init__(self)
        self.feedforward_y = Feedforward(n_layers, layer_width, dim_y, dim_added_y - dim_y)
        self.Koopman = nn.Linear(dim_y + dim_added_y, dim_y + dim_added_y, bias=False)
        self.dimState = dim_y + dim_added_y
        self.dim_y = dim_y
    
    def forward(self, y):
        sai_y = self.lift(y)
        y_next_approx = self.Koopman(sai_y)
        return y_next_approx
    
    def lift(self, y):
        y2 = self.feedforward_y(y)
        sai_y = torch.cat([y, y2], dim=1)
        return sai_y
    
    def koopmanOperate(self, sai_y):
        sai_y2y = self.Koopman(sai_y)
        return sai_y2y
    
    def getKoopmanOperator(self):
        Kyy = np.zeros([self.dimState, self.dimState])
        for param in self.Koopman.parameters(): # There is just one parameter, the Matrix of weights
            Kyy[:] = param[:].detach().numpy()
        return Kyy
        