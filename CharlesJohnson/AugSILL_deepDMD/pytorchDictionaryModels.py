import time
import numpy as np
from scipy.integrate import odeint
import random
import torch._C as torch 
from torch import nn as nn
from scipy.linalg import logm, expm
from scipy.integrate import odeint
import numpy as np
import matplotlib

# Support classes
class shiftedPolyoid(nn.Module):
    """ a polyoid is a function, p, of the form:
    p(x) = sum([steepness[i](x[i] - center[i])**power for i in range(len(x))]), where x is the a vector.
    So, for example, if x is 2D and the power is 2, the shifted polyoid is an arbitrary eliptical or 
    hyperbolic paraboloid.
    """
    def __init__(self, dim, power):
        super(shiftedPolyoid, self).__init__()
        self.dim = dim
        self.power = power
        self.centers = torch.nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = torch.nn.parameter.Parameter(torch.ones(dim))
        nn.init.uniform_(self.centers, 0, 10)
        nn.init.uniform_(self.steepnesses, 0, 10)
        
    def forward(self, x):
        pre_exponent = torch.sub(x, self.centers) # all these opperations are pointwise
        exponent = torch.pow(pre_exponent, self.power) 
        pre_sum = torch.mul(self.steepnesses, exponent)
        final_sum = torch.sum(pre_sum, axis=1)
        return final_sum
    
    def get_centers(self):
        return self.centers.detach().numpy()
    
    def get_steepnesses(self):
        return self.steepnesses.detach().numpy()
    
    
class polyoidDict(nn.Module):
    def __init__(self, dim, powers, numOfEachPower):
        """
        The polyoid dictionary is biased, state inclusive, and has polyoid basis elements.
        The variables, powers and numOfEachPower are lists of the same length. Powers has the powers for the polyoids, 
           numOfEachPower has the integer number of each of the polyoids to be included in the dictionary.
        """
        super(polyoidDict, self).__init__()
        self.dim = dim
        self.numNonlinear = sum(numOfEachPower)
        polysList = []
        for i in range(len(powers)):
            for _j in range(numOfEachPower[i]): # start variable with an underscore because it is not used.
                polysList.append(shiftedPolyoid(dim, powers[i]))
        self.polys = nn.ModuleList(polysList)
        self.koopDim = 1 + dim + len(polysList)
        self.Koopman = nn.Linear(self.koopDim, self.koopDim, bias=False)
        nn.init.kaiming_normal_(self.Koopman.weight) 
        
    def lift(self, x): 
        polyoids = torch.zeros([self.numNonlinear, len(x)])
        for i in range(self.numNonlinear):
            polyoids[i] = self.polys[i](x)
        x = torch.cat([torch.ones(1, len(x)), x.transpose(0, 1), polyoids], dim=0) 
        x = x.transpose(0, 1)
        return x
        
    def forward(self, x):
        x = self.lift(x)
        #print("lifted x", x)
        x = self.Koopman(x)
        return x
    
    def getKoopmanOperator(self):
        K = np.zeros([self.dim, self.koopDim])
        for param in self.Koopman.parameters(): #There is just one parameter, it's the Matrix of weights
            K[:] = param[:].detach().numpy()
        return K
    
    def getCenters(self):
        centers = []
        for poly in self.polys:
            centers.append(poly.get_centers())
        return centers
    
    def getSteepnesses(self):
        centers = []
        for poly in self.polys:
            centers.append(poly.get_steepnesses())
        return centers

class conjLog(nn.Module):
    def __init__(self, dim):
        super(conjLog, self).__init__()
        self.dim = dim
        self.centers = torch.nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = torch.nn.parameter.Parameter(torch.ones(dim))
        nn.init.uniform_(self.centers, 0, 10)
        nn.init.uniform_(self.steepnesses, 0, 10)
        
    def forward(self, x):
        pre_exponent = torch.sub(x, self.centers) # check if all these opperations are pointwise
        exponent = torch.mul(self.steepnesses, pre_exponent) # check if all these opperations are pointwise
        pre_den = torch.exp(exponent) # check if all these opperations are pointwise
        den = torch.ones(self.dim) + pre_den # check if all these opperations are pointwise
        individuals = torch.div(torch.ones(self.dim), den) # check if all these opperations are pointwise
        #print("individuals", individuals)
        product = torch.prod(individuals, axis=1)
        return product
    
    def get_centers(self):
        return self.centers.detach().numpy()
    
    def get_steepnesses(self):
        return self.steepnesses.detach().numpy()

class conjRBF(nn.Module):
    def __init__(self, dim):
        super(conjRBF, self).__init__()
        self.dim = dim
        self.centers = torch.nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = torch.nn.parameter.Parameter(torch.ones(dim))
        nn.init.uniform_(self.centers, 0, 10)
        nn.init.uniform_(self.steepnesses, 0, 10)
        
    def forward(self, x):
        pre_exponent = torch.sub(x, self.centers) # check if all these opperations are pointwise
        exponent = torch.mul(self.steepnesses, pre_exponent) # check if all these opperations are pointwise
        pre_pre_den = torch.exp(exponent) # check if all these opperations are pointwise
        pre_den = torch.ones(self.dim) + pre_pre_den # check if all these opperations are pointwise
        den = torch.mul(pre_den, pre_den)
        num = pre_pre_den
        individuals = torch.div(num, den) # check if all these opperations are pointwise
        product = torch.prod(individuals, axis=1)
        return product
    
    def get_centers(self):
        return self.centers.detach().numpy()
    
    def get_steepnesses(self):
        return self.steepnesses.detach().numpy()

class AugSILL(nn.Module):
    def __init__(self, dim, numLog, numRBF):
        super(AugSILL, self).__init__()
        self.dim = dim
        self.numLog = numLog
        self.numRBF = numRBF
        self.logs = nn.ModuleList([conjLog(dim) for i in range(numLog)])
        self.RBFs = nn.ModuleList([conjRBF(dim) for i in range(numRBF)])
        self.koopDim = 1 + dim + numLog + numRBF
        self.Koopman = nn.Linear(self.koopDim, self.koopDim, bias=False)
        nn.init.kaiming_normal_(self.Koopman.weight)
        
    def lift(self, x): 
        conjLogs = torch.zeros([self.numLog, len(x)])
        conjRBFs = torch.zeros([self.numRBF, len(x)])
        for i in range(self.numLog):
            conjLogs[i] = self.logs[i](x)
        for i in range(self.numRBF):
            conjRBFs[i] = self.RBFs[i](x)
        x = torch.cat([torch.ones(1, len(x)), x.transpose(0, 1), conjLogs, conjRBFs], dim=0) 
        x = x.transpose(0, 1)
        return x
        
    def forward(self, x):
        x = self.lift(x)
        #print("lifted x", x)
        x = self.Koopman(x)
        return x
    
    def getKoopmanOperator(self):
        K = np.zeros([self.dim, self.koopDim])
        for param in self.Koopman.parameters(): #There is just one parameter, it's the Matrix of weights
            K[:] = param[:].detach().numpy()
        return K
    
    def getLogCenters(self):
        centers = []
        for log in self.logs:
            centers.append(log.get_centers())
        return centers
    
    def getLogSteepnesses(self):
        centers = []
        for log in self.logs:
            centers.append(log.get_steepnesses())
        return centers
    
    def getRbfCenters(self):
        centers = []
        for rbf in self.RBFs:
            centers.append(rbf.get_centers())
        return centers
    
    def getRbfSteepnesses(self):
        centers = []
        for rbf in self.RBFs:
            centers.append(rbf.get_steepnesses())
        return centers
    
class SILL(nn.Module):
    def __init__(self, dim, numLog):
        super(SILL, self).__init__()
        self.dim = dim
        self.numLog = numLog
        self.logs = nn.ModuleList([conjLog(dim) for i in range(numLog)])
        self.koopDim = 1 + dim + numLog
        self.Koopman = nn.Linear(self.koopDim, self.koopDim, bias=False)
        nn.init.kaiming_normal_(self.Koopman.weight) 
        
    def lift(self, x): 
        conjLogs = torch.zeros([self.numLog, len(x)])
        for i in range(self.numLog):
            conjLogs[i] = self.logs[i](x)
        x = torch.cat([torch.ones(1, len(x)), x.transpose(0, 1), conjLogs], dim=0) 
        x = x.transpose(0, 1)
        return x
        
    def forward(self, x):
        x = self.lift(x)
        x = self.Koopman(x)
        return x
    
    def getKoopmanOperator(self):
        K = np.zeros([self.dim, self.koopDim])
        for param in self.Koopman.parameters(): #There is just one parameter, it's the Matrix of weights
            K[:] = param[:].detach().numpy()
        return K
    
    def getCenters(self):
        centers = []
        for log in self.logs:
            centers.append(log.get_centers())
        return centers
    
    def getSteepnesses(self):
        centers = []
        for log in self.logs:
            centers.append(log.get_steepnesses())
        return centers
