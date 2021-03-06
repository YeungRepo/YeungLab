import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set up some plotting settings
import matplotlib
font1 = {"family": 'normal', "weight": "bold", "size": 15}
font2 = {"family": 'normal', "weight": "bold", "size": 9}
matplotlib.rc('font', **font1)

DIM = 20 # Sets which data is used and the names of the output plots
dimStr = str(DIM)

# Helper functions:
def str2list(myString):
    """ Takes in a string of the form: '[n_1, n_2, ..., n_m]', 
    where n_i is a number, and returns the list: [n_1, n_2, ..., n_m].
    """
    myList = myString.split(",")
    myList[0] = myList[0][1:]
    myList[-1] = myList[-1][:-1]
    numList = [float(num) for num in myList]
    return numList

results = pd.read_csv("AllSysAllDicSGD{0}D_Feb2021.csv".format(dimStr)) 
"""results_most = pd.read_csv("AllSysAllDicSGD{0}D_Feb2021.csv".format(dimStr)) 
results_hermite = pd.read_csv("HermiteLegendreSGD{0}D_Feb2021.csv".format(dimStr))

# Put the results together
results = pd.concat([results_most, results_hermite])
"""
# Get baseline
dmdResults = pd.read_csv("DMD_Feb2021.csv")


# Separate by type
Metrics = list(results.columns)[-7:]
Systems = list(set(results.System)) 
#ystems.remove("glycolic oscillator") # We don't have this data for most of the algorithms...
Algorithms = ["SGD with SILL", "SGD with rbfs", "SGD with Legendre Polynomials", "SGD with Hermite Polynomials", "SGD with AugSILL"]
Kdims = list(set(results.Kdim))
seeds = list(set(results["Random Seed"]))

print("Metrics", Metrics)
print("Systems", Systems)
print("Algorithms", Algorithms)
print("Kdims", Kdims)
print("Random seeds", seeds)

for metric in Metrics:
    for sys in Systems:
        fig = plt.figure()
        ax = plt.subplot(111)
        for alg in Algorithms:
            # Grab the subset of the data that we want
            # First get the rows
            resPreRow = results.loc[results["System"]==sys]
            resRow = resPreRow.loc[resPreRow["Algorithm"]==alg]
            # Now get the column
            dataStrings = resRow[metric].to_numpy()
            dataBloc = np.array([str2list(stringData) for stringData in dataStrings])
            logDataBloc = np.log(dataBloc)
            # Get the mean and standard deviation
            means = np.mean(logDataBloc, axis=0)
            sds = np.std(logDataBloc, axis=0)
            # Build the plot of score vs error with error bars
            epochs = [50*i for i in range(len(means))]
            ax.errorbar(epochs, means, yerr=sds, label="sys={0}, alg={1}".format(sys, alg))
        # Put in the range of results for standard DMD
        dmd = dmdResults.loc[dmdResults["System"]==sys]
        dmdVals = dmd[metric].to_numpy()
        logDmdVals = np.log(dmdVals)
        dmdMean = np.mean(logDmdVals)
        dmdStd = np.std(logDmdVals)
        ax.plot(epochs, dmdMean * np.ones(len(epochs)), "b", alpha=0.5, label="DMD mean Performance")
        ax.plot(epochs, (dmdMean + dmdStd) * np.ones(len(epochs)), "b--", alpha=0.5)
        ax.plot(epochs, (dmdMean - dmdStd) * np.ones(len(epochs)), "b--", alpha=0.5)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Standardize the y-axis
        plt.ylim(-11, 10)
        if metric == "Time":
            matplotlib.rc('font', **font2)
            plt.legend(loc='center')
            matplotlib.rc('font', **font1)
        if metric == "Test Error5":
            plt.title("Training Epoch vs 5-step Log Test Error\n for system={0}".format(sys))
        else:
            plt.title("Training Epoch vs Log {0}\nFor sys={1}".format(metric, sys))
        fig.savefig("Plots/SGD_Metric={0}_sys={1}DimK={2}.jpg".format(metric, sys, dimStr))

