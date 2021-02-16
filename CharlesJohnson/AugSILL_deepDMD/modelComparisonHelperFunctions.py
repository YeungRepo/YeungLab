# Helper functions for training and testing a host of Koopman learning algorithms
# Written by Charles Johnson in 2021, charles.addisonj@gmail.com

import time
import pandas as pd
import numpy as np
from scipy.integrate import odeint
import random
import torch#._C as torch 
from torch import nn as nn
from scipy.linalg import logm, expm
from scipy.integrate import odeint
import numpy as np
import matplotlib
from pytorchDeepDMD_Models import feedforward_DDMD_no_input
from pytorchDictionaryModels import SILL_Model, AugSILL_Model, RBF_DictModel, hermiteDictModel, legendreDictModel


# Support functions.

def runEpochs(n_epochs, train_data, verify1_data, verify2_data, verify5_data, minibatch_size, koop_net, opt, loss,
             test_data, test_verify1_data, test_verify2_data, test_verify5_data, loss_val, samplerate):
    """
    Runs training epochs for a spcified module from pytorch. This model is learning a discrete-time state-space model.
      As it does so, it will check 1-step prediction error, 2-step prediction error and 5-step prediction error of the 
      data that it trains on and on an additional "test" data set.  The checks occur every samplerate epochs.
    2-step prediction error 
    (can be used to train neural networks or other constructs via SGD)
    @param n_epochs: int, the number of training epochs to run.
    @param train_data: 2D numpy array, the first dimension is system state dimension, the second one is time (probably should be shuffled).
    @param verify1_data: 2D numpy array, containing the next timesteps' of data for train_data.
    @param verify2_data: 2D numpy array, containing the data for two steps out from train_data.
    @param verify5_data: 2D numpy array, containing the data for five steps out from train_data.
    @param minibatch_size: the number of data points to put in each minibatch.
    @param koop_net: a pytorch module for Koopman learning, it needs to have a lift() method which takes the state and sends it to the higher 
        dimension specified by the dictionary of the koopman model.
    @param opt: the pytorch optimizer.
    @param loss: the pytorch loss function used for training.
    @param test_data: 2D numpy array, the first dimension is system state dimension, the second one is time (probably should be shuffled).
    @param test_verify1_data:  2D numpy array, containing the next timesteps' of data for test_data.
    @param test_verify2_data: 2D numpy array, containing the data for two steps out from test_data.
    @param test_verify5_data: 2D numpy array, containing the data for five steps out from test_data.
    @param loss_val: the pytorch loss function used for validation (this may be the same function as loss).
    @param samplerate: int, every samplerate epochs we record validation and diagnostic data.
    @returns: koop_net: our trained pytorch module, 
              train_errors1: a list of the recorded 1-step training errors, 
              train_errors2: a list of the recorded 2-step training errors, 
              train_errors5: a list of the recorded 5-step training errors, 
              test_errors1: a list of the recorded 1-step testing errors, 
              test_errors2: a list of the recorded 2-step testing errors, 
              test_errors5: a list of the recorded 5-step testing errors, 
              times: a list of the recorded amount of time taken to run.
    """
    train_errors1 = []
    train_errors2 = []
    train_errors5 = []
    test_errors1 = [] 
    test_errors2 = [] 
    test_errors5 = [] 
    times = []
    start = time.time()
    for epoch in range(n_epochs):
        indicies = [i for i in range(len(train_data[0]))]
        np.random.shuffle(indicies)
        train_data = train_data[:, indicies]
        verify1_data = verify1_data[:, indicies]
        verify2_data = verify2_data[:, indicies]
        verify5_data = verify5_data[:, indicies]
        num_minibatches = len(train_data[0]) // minibatch_size
        koop_net.train()
        for i in range(num_minibatches):
            datapoint = get_minibatch(train_data, minibatch_size, i)
            next_datapoint = get_minibatch(verify1_data, minibatch_size, i)
            next_datapoint2 = get_minibatch(verify2_data, minibatch_size, i)
            next_datapoint5 = get_minibatch(verify5_data, minibatch_size, i)
            datapoint = torch.tensor(datapoint).float()
            datapoint = datapoint.transpose(0, 1)
            next_datapoint = torch.tensor(next_datapoint).float()
            next_datapoint = next_datapoint.transpose(0, 1)
            
            next_datapoint2 = torch.tensor(next_datapoint2).float()
            next_datapoint2 = next_datapoint2.transpose(0, 1)
            
            next_datapoint5 = torch.tensor(next_datapoint5).float()
            next_datapoint5 = next_datapoint5.transpose(0, 1)
            opt.zero_grad()
            # put the right part of the datapoint into the neural network
            liftedAftr = koop_net.lift(next_datapoint)
            liftedAftr2 = koop_net.lift(next_datapoint2)
            liftedAftr5 = koop_net.lift(next_datapoint5)
            koop_approx = koop_net(datapoint) 
            error = loss(koop_approx, liftedAftr) 
            error.backward()
            opt.step()
            with torch.no_grad():
                #print(koop_approx.size())
                error2 = loss(koop_net.Koopman(koop_approx), liftedAftr2)
                error5 = loss(koop_net.Koopman(koop_net.Koopman(koop_net.Koopman(koop_net.Koopman(koop_approx)))),
                              liftedAftr5)
        if epoch % samplerate == 0:
            end = time.time()
            times.append(end - start)
            train_errors1.append(error.item())
            train_errors2.append(error2.item())
            train_errors5.append(error5.item())
            koop_net.eval()
            with torch.no_grad():
                # Do evaluation
                datapoint = torch.tensor(test_data).float()
                datapoint = datapoint.transpose(0, 1)
                next_datapoint = torch.tensor(test_verify1_data).float()
                next_datapoint = next_datapoint.transpose(0, 1)
                next2_datapoint = torch.tensor(test_verify2_data).float()
                next2_datapoint = next2_datapoint.transpose(0, 1)
                next5_datapoint = torch.tensor(test_verify5_data).float()
                next5_datapoint = next5_datapoint.transpose(0, 1)
                liftedAftr = koop_net.lift(next_datapoint)
                koop_approx = koop_net(datapoint)
                error1 = loss_val(koop_approx, liftedAftr)
                test_errors1.append(error1.item())
                liftedAftr = koop_net.lift(next2_datapoint)
                koop_approx = koop_net.Koopman(koop_approx)
                error2 = loss_val(koop_approx, liftedAftr)
                test_errors2.append(error2.item())
                liftedAftr = koop_net.lift(next5_datapoint)
                koop_approx = koop_net.Koopman(koop_net.Koopman(koop_net.Koopman(koop_approx)))
                error5 = loss_val(koop_approx, liftedAftr)
                test_errors5.append(error5.item())
    return koop_net, train_errors1, train_errors2, train_errors5, test_errors1, test_errors2, test_errors5, times

def getPowersAndNumbers(dim, added_dim):
    """ 
    This helper function takes int dim and int added_dim and returns a list of 
        integer polynomial orders (powers), along with a list of how many of each
        power (numOfEachPower).  These values are used to automatically generate 
        the search space of basis elements for polynomial dictionary models.
    """
    lenPowers = added_dim // dim
    leftover = added_dim % dim
    if leftover != 0:
        lenPowers += 1
    powers = [2 + i for i in range(lenPowers)]
    numOfEachPower = [dim] * lenPowers
    if leftover != 0:
        numOfEachPower[-1] = leftover
    return powers, numOfEachPower

def trainFFDDMD(train_data, verify1_data, verify2_data, verify5_data, test_data, test_verify1_data,
             test_verify2_data, test_verify5_data, 
             n_epochs=5000,
             dim=7,
             dim_added=20,
             n_layers=7, 
             layer_width=20,
             minibatch_size=30,
             samplerate=250, 
             parallel=False):
    """ 
    Function to do deepDMD to learn a Koopman model from timeseries data.
    @param train_data: 2D numpy array, the first dimension is system state dimension, the second one is time (probably should be shuffled).
    @param verify1_data: 2D numpy array, containing the next timesteps' of data for train_data.
    @param verify2_data: 2D numpy array, containing the data for two steps out from train_data.
    @param verify5_data: 2D numpy array, containing the data for five steps out from train_data.
    @param test_data: 2D numpy array, the first dimension is system state dimension, the second one is time (probably should be shuffled).
    @param test_verify1_data:  2D numpy array, containing the next timesteps' of data for test_data.
    @param test_verify2_data: 2D numpy array, containing the data for two steps out from test_data.
    @param test_verify5_data: 2D numpy array, containing the data for five steps out from test_data.
    @param n_epochs: int, the number of training epochs to run.
    @param dim: int, the dimension of the system that we used to generate our data.
    @param dim_added: int, the number of additional dimensions that our trained Koopman model will have.
    @param n_layers: int, the number of layers that our feedforward neural network for dictionary learning will have.
    @param layer_width: int, the width of each of the layers in the neural network.
    @param minibatch_size: the number of data points to put in each minibatch.
    @param samplerate: int, every samplerate epochs we record validation and diagnostic data.
    @param parallel: bool, do we try to run the computations in parallel, currently this does NOT work.  The code needs to be fixed. 
    @returns: koop_net: our trained pytorch module, 
              train_errors1: a list of the recorded 1-step training errors, 
              train_errors2: a list of the recorded 2-step training errors, 
              train_errors5: a list of the recorded 5-step training errors, 
              test_errors1: a list of the recorded 1-step testing errors, 
              test_errors2: a list of the recorded 2-step testing errors, 
              test_errors5: a list of the recorded 5-step testing errors, 
              times: a list of the recorded amount of time taken to run.
    """
    # Try this line below and see if it messes anything up...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = nn.MSELoss()
    loss_val = nn.MSELoss()
    koop_net = feedforward_DDMD_no_input(n_layers, layer_width, dim, dim_added)
    if parallel:
        koop_net = nn.DataParallel(koop_net)
        koop_net.to(device)
    opt = torch.optim.Adam(koop_net.parameters(), lr=0.025 , weight_decay=0.0001)
    koop_net, tEs1, tEs2, tEs5, TEs1, TEs2, TEs5, times = runEpochs(n_epochs, train_data, verify1_data, verify2_data, 
                                                                   verify5_data, minibatch_size, koop_net, opt, loss, 
                                                                   test_data, test_verify1_data, test_verify2_data, 
                                                                   test_verify5_data, loss_val, samplerate)
    train_errors1, train_errors2, train_errors5 = tEs1, tEs2, tEs5 
    test_errors1, test_errors2, test_errors5 = TEs1, TEs2, TEs5
    return [koop_net, train_errors1, train_errors2, train_errors5, 
            test_errors1, test_errors2, test_errors5, times]

# Function to train the neural network.
def trainSGD(train_data, verify1_data, verify2_data, verify5_data, test_data, test_verify1_data,
             test_verify2_data, test_verify5_data, 
            n_epochs = 5000,
            dim=2,
            added_logs=2,
            added_rbfs= 2,
            added_dim = 4,
            minibatch_size=30,
            samplerate=50,
            dictionary="AugSILL"):
    """
    Function to learn our parameters and test them via some variation on SGD.
    @param train_data: 2D numpy array, the first dimension is system state dimension, the second one is time (probably should be shuffled).
    @param verify1_data: 2D numpy array, containing the next timesteps' of data for train_data.
    @param verify2_data: 2D numpy array, containing the data for two steps out from train_data.
    @param verify5_data: 2D numpy array, containing the data for five steps out from train_data.
    @param test_data: 2D numpy array, the first dimension is system state dimension, the second one is time (probably should be shuffled).
    @param test_verify1_data:  2D numpy array, containing the next timesteps' of data for test_data.
    @param test_verify2_data: 2D numpy array, containing the data for two steps out from test_data.
    @param test_verify5_data: 2D numpy array, containing the data for five steps out from test_data.
    @param n_epochs: int, the number of training epochs to run.
    @param dim: int, the dimension of the system that we used to generate our data.
    @param added_logs: int, for SILL or AugSILL dictionaries, this is the number of conjunctive logistic dictionary elements.
    @param added_rbfs: int, for AugSILL dictionaries, this is the number of conjunctive logistic dictionary elements.
    @param added_dim: int, the number of additional dimensions that our trained Koopman model will have. For the AugSILL dictionary, 
        added_dim = added_logs + added_rbfs and for the SILL dictionary, added_dim = added_logs.
    @param minibatch_size: the number of data points to put in each minibatch.
    @param samplerate: int, every samplerate epochs we record validation and diagnostic data.
    @param dictionary: string, a prespecified label for the type of dictionary that our Koopman model is going to use. It can currently
        be one of the following: "rbf", "hermite", "legendre".  If it takes on any other value (including "AugSILL"), then the values of 
        added_logs and added_rbfs will be used to automatically determine which model is being used.
    @returns: koop_net: our trained pytorch module, 
              train_errors1: a list of the recorded 1-step training errors, 
              train_errors2: a list of the recorded 2-step training errors, 
              train_errors5: a list of the recorded 5-step training errors, 
              test_errors1: a list of the recorded 1-step testing errors, 
              test_errors2: a list of the recorded 2-step testing errors, 
              test_errors5: a list of the recorded 5-step testing errors, 
              times: a list of the recorded amount of time taken to run.
    """
    loss = nn.MSELoss()
    loss_val = nn.MSELoss()
    if added_rbfs < 1 and added_logs > 0:
        # use the regular SILL basis
        koop_net = SILL_Model(dim, added_logs)
    elif added_rbfs < 1 and added_logs < 1:
        if dictionary == "rbf":
            # use RBFs for dictionary elements
            koop_net = RBF_DictModel(dim, added_dim)
        elif dictionary == "hermite":
            # Use summed 1d hermite polynomials for dictionary elements
            powers, numOfEachPower = getPowersAndNumbers(dim, added_dim)
            koop_net = hermiteDictModel(dim, powers, numOfEachPower)
        elif dictionary == "legendre":
            # Use summed 1d legendre polynomials for dictionary elements
            powers, numOfEachPower = getPowersAndNumbers(dim, added_dim)
            koop_net = legendreDictModel(dim, powers, numOfEachPower)
    else:
        # use the augmented SILL basis
        koop_net = AugSILL_Model(dim, added_logs, added_rbfs)
    opt = torch.optim.Adam(koop_net.parameters(), lr=0.00015, weight_decay=0.0001)
    koop_net, tEs1, tEs2, tEs5, TEs1, TEs2, TEs5, times = runEpochs(n_epochs, train_data, verify1_data, verify2_data, 
                                                                   verify5_data, minibatch_size, koop_net, opt, loss, 
                                                                   test_data, test_verify1_data, test_verify2_data, 
                                                                   test_verify5_data, loss_val, samplerate)
    train_errors1, train_errors2, train_errors5 = tEs1, tEs2, tEs5 
    test_errors1, test_errors2, test_errors5 = TEs1, TEs2, TEs5
    return [koop_net, train_errors1, train_errors2, train_errors5, 
            test_errors1, test_errors2, test_errors5, times]

def get_Knew(error, lifting_func, datapoints):
    """
    Picks a column of the Koopman Operator to minimize, this is used in the matching pursuit algorithm.
    @param error: a m by n array, where m is the number of data points and n,
        is the system dimension, each row is the error at the data point in 
        the corresponding row of the function argument datapoints.
    @param lifting_func: a function that takes in a vector of length n, and 
        returns a scaler.
    @param datapoints: a m by n array, where m is the number of datapoints, 
        and n is the system dimension.
    returns: The constants to apply to the lifting function to minimize the error
        over the data points, and the error over the datapoints when the lifting factor
        is subtracted from the error surface.
    """
    m, n = np.shape(datapoints)
    lifting_vals = [lifting_func(datapoints[i]) for i in range(m)]
    A = np.vstack([np.eye(n) * lifting_vals[i] for i in range(m)])
    ATAinv = (np.sum(np.square(lifting_vals)))**-1 * np.eye(n)
    errors = error.reshape(m * n)
    Kcol = ATAinv @ A.transpose() @ errors
    return Kcol, np.linalg.norm(errors - A @ Kcol)


def matching_persuit(error, lifting_points, datapoints, niters):
    """
    Finds basis elements to try with EDMD by greedily selecting the best basis element
        one at a time from a set of availible ones.
    @param error: a m by n array, where m is the number of data points and n,
        is the system dimension, each row is the error at the data point in 
        the corresponding row of the function argument datapoints.
    @param lifting_funcs: a list of functions that take in a vector of length n, and 
        return a scaler.
    @param datapoints: a m by n array, where m is the number of datapoints, 
        and n is the system dimension.
    @param niters: The number of lifting points to iteratively subtract from the error
        surface.
    @returns: niters indicies of lifting_points.
    """
    error = np.copy(error)
    indicies = []
    allCols = []
    potential_indicies = [i for i in range(len(lifting_points))]
    for i in range(niters):
        best_error = np.inf
        best_col = -1
        best_index = -1
        for index in potential_indicies:
            colVals, error_val = get_Knew(error, lifting_points[index], datapoints)
            if error_val < best_error:
                best_error = error_val
                best_col = colVals
                best_index = index
        if best_index in potential_indicies:
            potential_indicies.remove(best_index)
            indicies.append(best_index)
            allCols.append(best_col)
            error -= datapoints @ np.diag(best_col)
        else:
            print("Index that not in list by iteration {0}:".format(str(i)), best_index)
            return indicies
    return indicies


def DMD_multi_trajectory(data):
    """ Approximates a Koopman generator using y as the observables given n data points of multiple system 
        trajectories.
    @param data: a 3d array of floats, the system trajectorys from y_0 to y_n. The first dimension has all the 
        separate trajectories. The second is the state dimension and the third is time.
    @returns: a 2d array of floats, the approximate koopman operator
    """
    num_traj, p, _n = np.shape(data)
    out = np.zeros([p, p])
    prev_data = data[0, :, 0:-1]
    fut_data = data[0, :, 1:]
    for t in range(1, num_traj):
        prev_data = np.hstack([prev_data, data[t, :, 0:-1]])
        fut_data = np.hstack([fut_data, data[t, :, 1:]])
    prev_pseudoinverse = np.linalg.pinv(prev_data)
    K_approx = fut_data @ prev_pseudoinverse
    out = K_approx
    return out


def EDMD_multi_trajectory(data, state_function):
    """ Approximates a Koopman generator using psi(y) as the observables given n data points 
        of multiple system trajectories.
    @param data: a 3d array of floats, the system trajectorys from y_0 to y_n. The first dimension has all the 
        separate trajectories. The second is the state dimension and the third is time.
    @param state_function: a function that takes in a 1D array (the current state vector), and returns a lifted 1D array 
        (the value of the lifted observables for that state vector).
    @returns: a 2d array of floats, the approximate koopman operator
    """
    num_traj, p, n = np.shape(data)
    out = np.zeros([p, p])
    prev_data = np.array([state_function(data[0, :, i]) for i in range(n-1)])
    fut_data = np.array([state_function(data[0, :, i+1]) for i in range(n-1)])
    for t in range(1, num_traj):
        prev_data = np.vstack([prev_data, np.array([state_function(data[t, :, i]) for i in range(n-1)])])
        fut_data = np.vstack([fut_data, np.array([state_function(data[t, :, i+1]) for i in range(n-1)])])
    #print(prev_data.transpose())
    prev_pseudoinverse = np.linalg.pinv(prev_data.transpose())
    K_approx = fut_data.transpose() @ prev_pseudoinverse
    out = K_approx
    return out


def logistic(x, steepness, position):
    return 1. / (1 + np.exp(steepness * (x - position)))

def logistic_prod(x_vec, steepness_vec, position_vec):
    prod = 1
    for i in range(len(x_vec)):
        prod *= logistic(x_vec[i], steepness_vec[i], position_vec[i])
    return prod

def rbf(x, steepness, position):
    exponent = np.exp(steepness * (x - position))
    return exponent / (1 + exponent)**2

def rbf_prod(x_vec, steepness_vec, position_vec):
    prod = 1
    for i in range(len(x_vec)):
        prod *= rbf(x_vec[i], steepness_vec[i], position_vec[i])
    return prod


def calc_data_Koop_error_MT(data, lifting_map, K):
    """
    Multi-Trajectory error calculation for a Koopman model on the specified data..
    @param data: a 3d array of floats, the system trajectorys from y_0 to y_n. The first dimension has all the 
        separate trajectories. The second dimension is the state and the third is time.
    @param lifting_map: a function that takes in a 1D array (the current state vector), and returns a lifted 1D array 
        (the value of the lifted observables for that state vector).
    @param K: a 2D numpy array of floats, the Koopman operator that works in conjunction with lifting_map.
    @returns: the norm of the 1-step, 2-step and 5-step prediction error of the Koopman model on this data.
    """
    data, data1, data2, data5 = format_data(data)
    p, num = np.shape(data)
    errors1 = []
    errors2 = []
    errors5 = []
    Krows, Kcols = np.shape(K)
    K_is_square = Krows == Kcols
    for col in range(num - 1):
        current = data[:, col]
        future1 = data1[:, col]
        future2 = data2[:, col]
        future5 = data5[:, col]
        lifted_guess1 = K @ lifting_map(current)
        if K_is_square:
            guess1 = lifted_guess1
        else:
            guess1 = lifted_guess1[1:p+1] # Here we assume that the first term is a bias state, not the measurement.
        lifted_guess2 = K @ lifted_guess1
        if K_is_square:
            guess2 = lifted_guess2
        else:
            guess2 = lifted_guess2[1:p+1] # Here we assume that the first term is a bias state, not the measurement.
        lifted_guess5 = K @ K @ K @ lifted_guess2
        if K_is_square:
            guess5 = lifted_guess5
        else:
            guess5 = lifted_guess5[1:p+1] # Here we assume that the first term is a bias state, not the measurement.
        error1 = future1 - guess1
        errors1.append(error1)
        error2 = future2 - guess2
        errors2.append(error2)
        error5 = future5 - guess5
        errors5.append(error5)
    return [np.linalg.norm(errors1) / num, np.linalg.norm(errors2) / num, 
            np.linalg.norm(errors5) / num] # Normalize for the number of things being measured.


# This version considers finite approximate closure
def get_Knew_expanded(error, lifting_func, datapoints, dim=2):
    """
    Minimizes the full Koopman operator, with past vals fixed given a new dictionary 
        element, this is used in the expanded matching pursuit algorithm.
    @param error: a m by n array, where m is the number of data points and n,
        is the augmented system dimension, each row is the error at the data point in 
        the corresponding row of the function argument datapoints.
    @param lifting_func: a function that takes in a vector of length n, and 
        returns a scaler.
    @param datapoints: a m by n array, where m is the number of datapoints, 
        and n is the augmented system dimension. Because this is augmented, only up to
        dim are the datapoints what we originally measured, each column past the dim'th 
        is actually a lifted dimension.
    returns: The constants to apply to the lifting function to minimize the error
        over the data points, and the constants to apply to all (including the added) 
        lifting functions to minimize the error of the lifing function iself and the 
        error over the datapoints when the lifting factor is subtracted from the error 
        surface.
    """
    m, n = np.shape(datapoints)
    lifting_vals = [lifting_func(datapoints[i][:dim]) for i in range(m)]
    A = np.zeros([(n + 1) * m, n + (n + 1)])
    for i in range(m):
        A[i*(n+1):(i+1)*(n+1)-1, 0:n] = np.eye(n) * lifting_vals[i] # Addition to past error dims
        # Added error dim for original functions
        A[(i+1)*(n+1)-1, n:-1] = datapoints[i]
        # Added error dim for the new lifting function.
        A[(i+1)*(n+1)-1, -1] = lifting_vals[i] 
    spaced_error = np.hstack([error, np.zeros([len(error), 1])])
    errors = spaced_error.reshape(m * (n + 1))
    Kvals = np.linalg.lstsq(A, errors, rcond=None)[0]
    Kcol = Kvals[:n]
    Krow = Kvals[n:]
    return Kcol, Krow, np.linalg.norm(errors - A @ Kvals)

def expanded_matching_persuit(error, K, lifting_points, datapoints, niters):
    """
    Finds basis elements to try with EDMD, by greedily adding them one at a time.
    @param error: a m by n array, where m is the number of data points and n,
        is the system dimension, each row is the error at the data point in 
        the corresponding row of the function argument datapoints.
    @param K0: the Koopman operator learned from DMD.
    @param lifting_points: a list of z functions that take in a vector of length n, and 
        return a scaler.
    @param datapoints: a m by n array, where m is the number of datapoints, 
        and n is the system dimension.
    @param niters: The number of lifting points to iteratively subtract from the error
        surface.
    @returns: niters indicies of lifting_points.
    """
    m, n_orig = np.shape(datapoints)
    error = np.copy(error)
    indicies = []
    potential_indicies = [i for i in range(len(lifting_points))]
    for _i in range(niters):
        best_error = np.inf
        best_col, best_row, best_index = -1, -1, -1
        for index in potential_indicies:
            #print("arguments to getKnewExpanded", error, lifting_points[index], 
            #     past_lifting_funcs, datapoints)
            colVals, rowVals, err = get_Knew_expanded(error, lifting_points[index], datapoints, n_orig)
            if err < best_error:
                best_error = err
                best_col = colVals
                best_row = rowVals
                best_index = index
        #print("arguments to K_top: ", K, np.array([last_col]).transpose(), last_col.reshape(len(last_col), 1))
        K_top = np.hstack([K, best_col.reshape(len(best_col), 1)])
        K = np.vstack([K_top, best_row])
        potential_indicies.remove(best_index)
        indicies.append(best_index)
        # Update the error:
        lifted_datapoints = [lifting_points[best_index](datapoint[:n_orig]) for datapoint in datapoints]
        m, n = np.shape(datapoints)
        lifted_block = np.ones([m, n])
        for j in range(m):
            lifted_block[j] *= lifted_datapoints[j]
        error -= lifted_block @ np.diag(best_col)
        new_error_dim = [np.dot(best_row[:-1], datapoints[j]) + best_row[-1] * lifted_datapoints[j] 
                         for j in range(m)]
        new_error_dim = np.array([new_error_dim]).transpose()
        error = np.hstack([error, new_error_dim])
        lifted_datapoints = np.array([lifted_datapoints]).transpose()
        datapoints = np.hstack([datapoints, lifted_datapoints])
    return indicies


# A function to train and test deepDMD with a feedforward neural network.
def train_and_test_FFdeepDMD(training_data, testing_data, dimK):
    """
    Trains and tests a feedforward deepDMD Koopman model on the training_data and then returns
    a number of values relevant to the performance of that model. This function is designed to 
    be used in an apples to apples comparison with the train_and_test_SGD() function.
    @param training_data: a 3d array of floats, the training system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @param testing_data: a 3d array of floats, the testing system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @param dimK: int, the dimension of the final Koopman operator.
    @returns: time taken, error norm for 1-step predictions on the training data, and
        error norm for 1-step predictions on the training data.
    """
    # Get data in the needed format
    train, train_verify1, train_verify2, train_verify5 = format_data(training_data)
    test, test_verify1, test_verify2, test_verify5 = format_data(testing_data)
    n, _m = np.shape(train)
    # run the training with the correct parameters and
    # run the testing as we train
    _koop_net, trainE1, trainE2, trainE5, testE1, testE2, testE5, times = trainFFDDMD(train, train_verify1, 
            train_verify2, train_verify5, test, test_verify1, test_verify2, test_verify5, 
             n_epochs = 5000,
             dim=n,
             dim_added=dimK - n,
             n_layers=7, 
             layer_width=20,
             minibatch_size=30,
             samplerate=250)
    # Grab the parameters for the centers and steepnesses from the network, so we can pass them in to 
    # the matching pursuit algorithm
    return [times, trainE1, trainE2, trainE5, testE1, testE2, testE5]

# We need a function to train and test SGD, 
# Its inputs should be training data, testing data, and dimension of the final operator
# it should return time taken, error norm for 1-step predictions.

def train_and_test_SGD(training_data, testing_data, dimK, dictionary="AugSILL"):
    """
    Trains and tests a  Koopman model on the training_data and then returns
    a number of values relevant to the performance of that model. This function is designed to 
    be used in an apples to apples comparison with the train_and_test_FFdeepDMD() function.
    @param training_data: a 3d array of floats, the training system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @param testing_data: a 3d array of floats, the testing system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @param dimK: int, the dimension of the final Koopman operator.
    @param dictionary: string, a prespecified label for the type of dictionary that our Koopman model is going to use. It can currently
        be one of the following: "rbf", "hermite", "legendre", "AugSILL" or "SILL".
    @returns: time taken, error norm for 1-step predictions on the training data, and
        error norm for 1-step predictions on the training data.
    """
    # Get data in the needed format
    train, train_verify1, train_verify2, train_verify5 = format_data(training_data)
    test, test_verify1, test_verify2, test_verify5 = format_data(testing_data)
    n, _m = np.shape(train)
    dim_added = dimK - n - 1
    if dictionary=="AugSILL":
        if dim_added % 2 == 0:
            n_logs, n_rbfs = dim_added // 2, dim_added // 2
        else:
            n_logs, n_rbfs = dim_added // 2 + 1, dim_added // 2
    elif dictionary=="SILL":
        n_logs, n_rbfs = dim_added, 0
    else:
        n_logs, n_rbfs = 0, 0
    # run the training with the correct parameters and
    # run the testing as we train
    _koop_net, trainE1, trainE2, trainE5, testE1, testE2, testE5, times = trainSGD(train, train_verify1, train_verify2, 
                                                 train_verify5, test, test_verify1, test_verify2, test_verify5,
                                                 dim=n, added_logs=n_logs, added_rbfs=n_rbfs, added_dim=dim_added, 
                                                 dictionary=dictionary)
    # Grab the parameters for the centers and steepnesses from the network, so we can pass them in to 
    # the matching pursuit algorithm
    return [times, trainE1, trainE2, trainE5, testE1, testE2, testE5]


def train_and_test_SGD_old(training_data, testing_data, dimK):
    """
    Trains and tests a Koopman model via SGD on the training_data and then returns
    a number of values relevant to the performance of that model. This function is designed to 
    be used in an apples to apples comparison with the train_and_test_MP_simple() and 
    train_and_test_MP_full() functions.
    @param training_data: a 3d array of floats, the training system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @param testing_data: a 3d array of floats, the testing system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @param dimK: int, the dimension of the final Koopman operator.
    @returns: time taken, error norm for 1-step predictions on the training data, and
        error norm for 1-step predictions on the training data.
    """
    # Get data in the needed format
    train, train_verify1, train_verify2, train_verify5 = format_data(training_data)
    test, test_verify1, test_verify2, test_verify5 = format_data(testing_data)
    n, _m = np.shape(train)
    dim_added = dimK - n - 1
    if dim_added % 2 == 0:
        n_logs, n_rbfs = dim_added // 2, dim_added // 2
    else:
        n_logs, n_rbfs = dim_added // 2 + 1, dim_added // 2
    # run the training with the correct parameters and
    # run the testing as we train
    koop_net, trainE1, trainE2, trainE5, testE1, testE2, testE5, times = trainSGD(train, train_verify1, train_verify2, 
                                                 train_verify5, test, test_verify1, test_verify2, test_verify5,
                                                 dim=n, added_logs=n_logs, added_rbfs=n_rbfs)
    # Grab the parameters for the centers and steepnesses from the network, so we can pass them in to 
    # the matching pursuit algorithm
    try: # This is to grab the centers and steepnesses when desired to pass in to Matching Pursuit for a specific test.
        logCenters = koop_net.getLogCenters()
        logSteeps = koop_net.getLogSteepnesses()
        rbfCenters = koop_net.getRbfCenters()
        rbfSteeps = koop_net.getRbfSteepnesses()
        return [times[-1], trainE1[-1], trainE2[-1], trainE5[-1], testE1[-1], testE2[-1], testE5[-1], logCenters, logSteeps, 
                rbfCenters, rbfSteeps]
    except AttributeError:
        return [times[-1], trainE1[-1], trainE2[-1], trainE5[-1], testE1[-1], testE2[-1], testE5[-1]]


# The function below grabs the baseline for training given by DMD
def train_and_test_DMD(training_data, testing_data):
    """This function will run standard DMD on the system (using all dynamic nodes, AKA no model reduction).
    This is intended to be used as a baseline which all other algorithms can improve upon.
    @param training_data: a 3d array of floats, the training system trajectories. 
        The first dimension has all the separate trajectories. 
    @param testing_data: a 3d array of floats, the testing system trajectories. 
        The first dimension has all the separate trajectories. The second is system state and 
        the third is time.
    @returns: time taken, the 2-norm of the error for 1-step, 2-step and 5-step predictions on the training data, 
         and the 2-norm of the error for 1-step, 2-step and 5-step predictions on the testing data.
    """
    # Set up for our run
    start = time.time()
    K = DMD_multi_trajectory(training_data)
    state_function = lambda x: x
    train_error1, train_error2, train_error5  = calc_data_Koop_error_MT(training_data, state_function, K)
    test_error1, test_error2, test_error5 = calc_data_Koop_error_MT(testing_data, state_function, K)
    end = time.time()
    return [end - start, train_error1, train_error2, train_error5, test_error1, test_error2, test_error5]


# We need a function to train and test Matching Pursuit, 
# Its inputs should be training data, testing data, steepness vals, position vals, and dimension of the final operator
# it should return time taken, error norm for 1-step predictions, total gridsize.

######## ########### ########### ########### ######## ########### ########### ###########
######## REMBER FOR PAPER ###########
# NOTE: For Matching Pursuit, we search over all rbfs and logs, whereas in SDG we specify an equal number of logs and 
# rbfs. This means that matching persuit is looking for a more optimal combination, at the price of time taken.

# Also, note that the time measurement is only vaguely helpful as some differences in time may or may not be
# accounted for by how I coded things up (like different packages/functions being used to calculate error...)
######## REMBER FOR PAPER ###########
######## ########### ########### ########### ######## ########### ########### ###########

def train_and_test_MP_simple(training_data, testing_data, steep_vals, pos_vals, dimK):
    """
    Trains and tests a Koopman model via MP on the training_data and then returns
    a number of values relevant to the performance of that model. This function is designed to 
    be used in an apples to apples comparison with the train_and_test_SGD_old() function.
    @param training_data: a 3d array of floats, the training system trajectories. 
        The first dimension has all the separate trajectories.
    @param testing_data: a 3d array of floats, the testing system trajectories. 
        The first dimension has all the separate trajectories.
    @param steep_vals: a list of possible steepnesses (each element in the list should have the 
        same number of dimensions as the system)
    @param pos_vals: a list of possible centers (each element in the list should have the 
        same number of dimensions as the system)
    @param dimK: int, the dimension of the final Koopman operator.
    @returns: time taken, error norm for 1-step predictions on the training data, 
        error norm for 1-step predictions on the training data, and total basis elements (gridsize).
    """
    # Set up for our run
    K = DMD_multi_trajectory(training_data)
    train, train_verify1, _train_verify2, _train_verify5 = format_data(training_data)
    datapoints = train.transpose()
    error = train_verify1.transpose() - datapoints @ K
    error -= np.mean(error) # automatically put in bias first
    # Get the possible basis elements
    basis_elements = []
    for s_val in steep_vals:
        for p_val in pos_vals:
            basis_elements.append(lambda x: logistic_prod(x, s_val, p_val))
            basis_elements.append(lambda x: rbf_prod(x, s_val, p_val))
    # Choose niters
    _m, n = np.shape(datapoints)
    niters = dimK - n - 1
    # run matching persuit
    start = time.time()
    ind_MP_simple = matching_persuit(error, basis_elements, datapoints, niters)
    # Get the train and testing error
    state_function = lambda x: [1.] + [x[i] for i in range(n)] + [basis_elements[ind](x) for ind in ind_MP_simple]
    K_simple = EDMD_multi_trajectory(training_data, state_function)
    train_error1, train_error2, train_error5  = calc_data_Koop_error_MT(training_data, state_function, K_simple)
    test_error1, test_error2, test_error5 = calc_data_Koop_error_MT(testing_data, state_function, K_simple)
    end = time.time()
    return [end - start, train_error1, train_error2, train_error5, test_error1, test_error2, test_error5,
            len(basis_elements)]
    

# We need a function to train and test Matching Pursuit, 
# Its inputs should be training data, testing data, steepness vals, position vals, and dimension of the final operator
# it should return time taken, error norm for 1-step predictions, total gridsize.

def train_and_test_MP_full(training_data, testing_data, steep_vals, pos_vals, dimK):
    """
    Trains and tests a Koopman model via MP on the training_data and then returns
    a number of values relevant to the performance of that model. This function is designed to 
    be used in an apples to apples comparison with the train_and_test_SGD_old() function.
    This function tends to do better than train_and_test_MP_simple(), but it takes longer to run.
    @param training_data: a 3d array of floats, the training system trajectories. 
        The first dimension has all the separate trajectories.
    @param testing_data: a 3d array of floats, the testing system trajectories. 
        The first dimension has all the separate trajectories.
    @param steep_vals: a list of possible steepnesses (each element in the list should have the 
        same number of dimensions as the system)
    @param pos_vals: a list of possible centers (each element in the list should have the 
        same number of dimensions as the system)
    @param dimK: int, the dimension of the final Koopman operator.
    @returns: time taken, error norm for 1-step predictions on the training data, 
        error norm for 1-step predictions on the training data, and total basis elements (gridsize).
    """
    # Set up for our run
    K = DMD_multi_trajectory(training_data)
    train, train_verify1, _train_verify2, _train_verify5 = format_data(training_data)
    datapoints = train.transpose()
    _m, n = np.shape(datapoints)
    error = train_verify1.transpose() - datapoints @ K
    error -= np.mean(error) # automatically put in bias first
    # Get the possible basis elements
    basis_elements = []
    for s_val in steep_vals:
        for p_val in pos_vals:
            basis_elements.append(lambda x: logistic_prod(x, s_val, p_val))
            basis_elements.append(lambda x: rbf_prod(x, s_val, p_val))
    # Choose niters
    niters = dimK - n - 1
    # run matching persuit
    start = time.time()
    ind_MP_full = expanded_matching_persuit(error, K, basis_elements, datapoints, niters)
    # Get the train and testing error
    state_function = lambda x: [1.] + [x[i] for i in range(n)] + [basis_elements[ind](x) for ind in ind_MP_full]
    K_full = EDMD_multi_trajectory(training_data, state_function)
    train_error1, train_error2, train_error5 = calc_data_Koop_error_MT(training_data, state_function, K_full)
    test_error1, test_error2, test_error5 = calc_data_Koop_error_MT(testing_data, state_function, K_full)
    end = time.time()
    return [end - start, train_error1, train_error2, train_error5,
            test_error1, test_error2, test_error5, len(basis_elements)]
    


# Getting our initial data via numerical simulation
def simulate(system, y0, times):
    """ Runs a system simulation.
    @param system: function of an array of floats, a function that takes in the state and 
        returns a new state (some unforced dynamic system).
    @param y0: array of floats, the starting state for our simulation.
    @param times: array of floats, the timesteps to run the simulation for.
    @returns: a 2d array of floats, the system trajectory from y_0 to y_time.
    """
    out = odeint(system, y0, times)
    return out.transpose()

def train_test_data_flexDim(sys, dim, nTrain=10, nTest=5, timeframe=5, 
                                                timesteps=101, 
                                                train_lims=[0, .5], 
                                                test_lims=[0, 1]):
    """
    Generates train and test data for a system. The data is simulated from time=0 to time=timeframe.
    @param sys: a function representing a dynamic systems whose arguments are, state x (1d array) 
        and time, t (float).
    @param dim: int, the dimension of the system.
    @param nTrain: int, the number of simulations to run for training data (each gets thier own, 
        randomly determined initial conditions).
    @param nTest: int, the number of simulations to run for testing data (each gets thier own, 
        randomly determined initial conditions).
    @param timeframe: float, how far past zero time goes during the simulations (should be positive).
    @param timesteps: int, the number of timepoints at which the data is explicitly measured.
    @param train_lims: tuple of floats, The limits from which train initial conditions are randomly 
        (uniformally) chosen.
    @param test_lims: tuple of floats, The limits from which test initial conditions are randomly 
        (uniformally) chosen.
    @returns: 2 3D numpy arrays, the first dimention is trajectory, the second state and the third time.
        One is for training data and the second is for testing data.
    """
    datas = []
    for _i in range(nTrain):
        ic = np.zeros(dim)
        for j in range(dim):
            ic[j] = np.random.uniform(train_lims[0], train_lims[1])
        datas.append(simulate(sys, ic, np.linspace(0, timeframe, timesteps)))
    train_data = np.array(datas)
    datas = []
    for _i in range(nTest):
        ic = np.zeros(dim)
        for j in range(dim):
            ic[j] = np.random.uniform(test_lims[0], test_lims[1])
        datas.append(simulate(sys, ic, np.linspace(0, timeframe, timesteps)))
    test_data = np.array(datas)
    return train_data, test_data

def train_test_data(sys, timeframe=15, timesteps=101, train_lims=[0, .5], test_lims=[0, 1]):
    """
    Generates train and test data for a system. We generate train data from 100 uniformally sampled
        initial conditions and test data from 50 uniformally sampled initial conditions.
    This only works for a 2D system.
    The data is simulated from time=0 to time=timeframe.
    @param sys: a function representing a dynamic systems whose arguments are, state x (1d array) 
        and time, t (float).
    @param timeframe: float, how far past zero time goes during the simulations (should be positive).
    @param timesteps: int, the number of timepoints at which the data is explicitly measured.
    @param train_lims: tuple of floats, The limits from which train initial conditions are randomly 
        (uniformally) chosen.
    @param test_lims: tuple of floats, The limits from which test initial conditions are randomly 
        (uniformally) chosen.
    @returns: 2 3D numpy arrays, the first dimention is trajectory, the second state and the third time.
        One is for training data and the second is for testing data.
    """
    datas = []
    for _i in range(100):
        ic = np.zeros(2)
        ic[0] = np.random.uniform(train_lims[0], train_lims[1])
        ic[1] = np.random.uniform(train_lims[0], train_lims[1])
        datas.append(simulate(sys, ic, np.linspace(0, timeframe, timesteps)))
    train_data = np.array(datas)
    datas = []
    for _i in range(50):
        ic = np.zeros(2)
        ic[0] = np.random.uniform(test_lims[0], test_lims[1])
        ic[1] = np.random.uniform(test_lims[0], test_lims[1])
        datas.append(simulate(sys, ic, np.linspace(0, timeframe, timesteps)))
    test_data = np.array(datas)
    return train_data, test_data


def get_center_vals(min_val, max_val, num=8):
    """ Grabs evenly spaced points on a 2D grid for use with MP algorithms.
    The grid's dimensions and mesh are determined by min_val, max_val and num.
    A list of 2-Tuples with the gridpoints in the square is returned.
    """
    centers = []
    vals = np.linspace(min_val, max_val, num)
    for i in range(num):
        for j in range(num):
            centers.append([vals[i], vals[j]])
    return centers


    # Function to format the data:
def format_data(data):
    """
    Function to take data, shuffle it and put it into a format for simple learning.
    In its final format, the data is 2 dimensional, the first dimension is the system state dimension, the second is time.  
    All the unique trajecotries will have been mixed together.
    @param data: 3D numpy array of floats, the first dimension is trajectories (each has its own initial conditions), 
        the second dimension is the system state dimension, the third dimension is time.
    @returns: 4 2D numpy arrays of floats: shuffled state data, the next timestep for all the shuffled data, 2 timesteps 
        out for the shuffled data and 5 timesteps out for the shuffled data.
    """
    pre_shuffle_data = np.hstack([data[i, :, :-5] for i in range(len(data))])
    pre_shuffle_verify1_data = np.hstack([data[i, :, 1:-4] for i in range(len(data))])
    pre_shuffle_verify2_data = np.hstack([data[i, :, 2:-3] for i in range(len(data))])
    pre_shuffle_verify5_data = np.hstack([data[i, :, 5:] for i in range(len(data))])
    
    indicies = [i for i in range(len(pre_shuffle_data[0]))] 
    np.random.shuffle(indicies)
    shuffled_data = pre_shuffle_data[:, indicies]
    shuffled_verify1_data = pre_shuffle_verify1_data[:, indicies]
    shuffled_verify2_data = pre_shuffle_verify2_data[:, indicies]
    shuffled_verify5_data = pre_shuffle_verify5_data[:, indicies]
    return shuffled_data, shuffled_verify1_data, shuffled_verify2_data, shuffled_verify5_data
    
def get_minibatch(data, minibatch_size, minibatch_number):
    """
    Helper function for the runEpochs() function, it just extracts the relevant minibatch of data.
    """
    ms, mn = minibatch_size, minibatch_number
    return data[:, mn*ms:(mn+1)*ms]