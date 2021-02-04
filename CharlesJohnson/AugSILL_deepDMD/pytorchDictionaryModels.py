import time
import numpy as np
from scipy.integrate import odeint
import random
from torch import nn
import torch#._C as torch 
from scipy.linalg import logm, expm
from scipy.integrate import odeint
import matplotlib

# The first 5 hermite polynomials
HERMITE_BASIS = [[1],
                 [0, 1],
                 [-1, 0, 1],
                 [0, -3, 0, 1],
                 [3, 0, -6, 0, 1],
                 [0, 15, 0, -10, 0, 1],
                 [-15, 0, 45, 0, -15, 0, 1]]

LEGENDRE_BASIS = [[1],
                 [0, 1],
                 [-1/2, 0, 3/2],
                 [0, -3/2, 0, 5/2],
                 [3/8, 0, -30/8, 0, 35/8],
                 [0, 15/8, 0, -70/8, 0, 63/8],
                 [-5/16, 0, 105/16, 0, -315/16, 0, 231/16]]

class legendrePoly(nn.Module):
    """ A sum of legendre polynomials of a given degree.
    """
    def __init__(self, dim, degree):
        super(legendrePoly, self).__init__()
        self.dim = dim
        try:
            self.degree = degree
        except IndexError:
            print("WARNING: IndexError! Your Degree is too high.")
        self.basis = LEGENDRE_BASIS[degree]
        self.centers = nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = nn.parameter.Parameter(torch.ones(dim))
        nn.init.uniform_(self.centers, 0, 10)
        nn.init.uniform_(self.steepnesses, 0, 10)
        
    def forward(self, x):
        poly_arg = torch.sub(x, self.centers) # all these opperations are pointwise
        poly = self.calc_legendre(poly_arg)
        weight_poly = torch.mul(poly, self.steepnesses)
        total_sum = torch.sum(weight_poly, axis=1)
        return total_sum

    def calc_legendre(self, x):
        x_shape = list(x.size())
        xs = torch.zeros([len(self.basis)] + x_shape)
        for i, coeff in enumerate(self.basis):
            xs[i] = torch.mul(x, coeff)
        poly_vals = torch.sum(xs, axis=0)
        return poly_vals
    
    def get_centers(self):
        return self.centers.detach().numpy()
    
    def get_steepnesses(self):
        return self.steepnesses.detach().numpy()


class legendreDict(nn.Module):
    def __init__(self, dim, powers, numOfEachPower):
        """
        The legendre dictionary is biased, state inclusive, and has basis elements that are sums of matching degree 
         legendre polynomials. The variables, powers and numOfEachPower are lists of the same length. Powers has the 
         powers for the legendre polynomials, numOfEachPower has the integer number of each of the polynomials to be 
         included in the dictionary.
        """
        super(legendreDict, self).__init__()
        self.dim = dim
        self.numNonlinear = sum(numOfEachPower)
        polysList = []
        for i in range(len(powers)):
            for _j in range(numOfEachPower[i]): # start variable with an underscore because it is not used.
                polysList.append(legendrePoly(dim, powers[i]))
        self.polys = nn.ModuleList(polysList)
        self.koopDim = 1 + dim + len(polysList)
        self.Koopman = nn.Linear(self.koopDim, self.koopDim, bias=False)
        nn.init.kaiming_normal_(self.Koopman.weight) 
        
    def lift(self, x): 
        polys = torch.zeros([self.numNonlinear, len(x)])
        for i in range(self.numNonlinear):
            polys[i] = self.polys[i](x)
        x = torch.cat([torch.ones(1, len(x)), x.transpose(0, 1), polys], dim=0) 
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
        for poly in self.polys:
            centers.append(poly.get_centers())
        return centers
    
    def getSteepnesses(self):
        centers = []
        for poly in self.polys:
            centers.append(poly.get_steepnesses())
        return centers


class hermitePoly(nn.Module):
    """ A sum of hermite polynomials of a given degree.
    """
    def __init__(self, dim, degree):
        super(hermitePoly, self).__init__()
        self.dim = dim
        try:
            self.degree = degree
        except IndexError:
            print("WARNING: IndexError! Your Degree is too high.")
        self.basis = HERMITE_BASIS[degree]
        self.centers = nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = nn.parameter.Parameter(torch.ones(dim))
        nn.init.uniform_(self.centers, 0, 10)
        nn.init.uniform_(self.steepnesses, 0, 10)
        
    def forward(self, x):
        poly_arg = torch.sub(x, self.centers) # all these opperations are pointwise
        poly = self.calc_hermite(poly_arg)
        weight_poly = torch.mul(poly, self.steepnesses)
        total_sum = torch.sum(weight_poly, axis=1)
        return total_sum

    def calc_hermite(self, x):
        x_shape = list(x.size())
        xs = torch.zeros([len(self.basis)] + x_shape)
        for i, coeff in enumerate(self.basis):
            xs[i] = torch.mul(x, coeff)
        poly_vals = torch.sum(xs, axis=0)
        return poly_vals
    
    def get_centers(self):
        return self.centers.detach().numpy()
    
    def get_steepnesses(self):
        return self.steepnesses.detach().numpy()


class hermiteDict(nn.Module):
    def __init__(self, dim, powers, numOfEachPower):
        """
        The hermite dictionary is biased, state inclusive, and has basis elements that are sums of matching degree 
         hermite polynomials. The variables, powers and numOfEachPower are lists of the same length. Powers has the 
         powers for the hermite polynomials, numOfEachPower has the integer number of each of the polynomials to be 
         included in the dictionary.
        """
        super(hermiteDict, self).__init__()
        self.dim = dim
        self.numNonlinear = sum(numOfEachPower)
        polysList = []
        for i in range(len(powers)):
            for _j in range(numOfEachPower[i]): # start variable with an underscore because it is not used.
                polysList.append(hermitePoly(dim, powers[i]))
        self.polys = nn.ModuleList(polysList)
        self.koopDim = 1 + dim + len(polysList)
        self.Koopman = nn.Linear(self.koopDim, self.koopDim, bias=False)
        nn.init.kaiming_normal_(self.Koopman.weight) 
        
    def lift(self, x): 
        polys = torch.zeros([self.numNonlinear, len(x)])
        for i in range(self.numNonlinear):
            polys[i] = self.polys[i](x)
        x = torch.cat([torch.ones(1, len(x)), x.transpose(0, 1), polys], dim=0) 
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
        for poly in self.polys:
            centers.append(poly.get_centers())
        return centers
    
    def getSteepnesses(self):
        centers = []
        for poly in self.polys:
            centers.append(poly.get_steepnesses())
        return centers


class rbf(nn.Module):
    """ a radial basis function
    """
    def __init__(self, dim):
        super(rbf, self).__init__()
        self.dim = dim
        self.centers = nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = nn.parameter.Parameter(torch.ones(dim))
        nn.init.uniform_(self.centers, 0, 10)
        nn.init.uniform_(self.steepnesses, 0, 10)
        
    def forward(self, x):
        pre_exponent = torch.sub(x, self.centers) # all these opperations are pointwise
        exponent = torch.mul(self.steepnesses, pre_exponent) 
        pre_pre_den = torch.exp(exponent) 
        pre_den = torch.ones(self.dim) + pre_pre_den 
        den = torch.mul(pre_den, pre_den)
        num = pre_pre_den
        individuals = torch.div(num, den) 
        total_sum = torch.sum(individuals, axis=1)
        return total_sum
    
    def get_centers(self):
        return self.centers.detach().numpy()
    
    def get_steepnesses(self):
        return self.steepnesses.detach().numpy()
    
    
class RBF_Dict(nn.Module):
    def __init__(self, dim, numRBF):
        """
        The polyoid dictionary is biased, state inclusive, and has polyoid basis elements.
        The variables, powers and numOfEachPower are lists of the same length. Powers has the powers for the polyoids, 
           numOfEachPower has the integer number of each of the polyoids to be included in the dictionary.
        """
        super(RBF_Dict, self).__init__()
        self.dim = dim
        self.numRBF = numRBF
        self.RBFs = nn.ModuleList([rbf(dim) for i in range(numRBF)])
        self.koopDim = 1 + dim + numRBF
        self.Koopman = nn.Linear(self.koopDim, self.koopDim, bias=False)
        nn.init.kaiming_normal_(self.Koopman.weight)
        
    def lift(self, x): 
        rbfs = torch.zeros([self.numRBF, len(x)])
        for i in range(self.numRBF):
            rbfs[i] = self.RBFs[i](x)
        x = torch.cat([torch.ones(1, len(x)), x.transpose(0, 1), rbfs], dim=0) 
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
        self.centers = nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = nn.parameter.Parameter(torch.ones(dim))
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
        self.centers = nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = nn.parameter.Parameter(torch.ones(dim))
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
        self.centers = nn.parameter.Parameter(torch.ones(dim))
        self.steepnesses = nn.parameter.Parameter(torch.ones(dim))
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
