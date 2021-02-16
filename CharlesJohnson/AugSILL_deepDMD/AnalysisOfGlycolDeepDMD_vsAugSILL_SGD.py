import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Set up some plotting settings
import matplotlib
font1 = {"family": 'normal', "weight": "bold", "size": 12}
matplotlib.rc('font', **font1)

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

results = pd.read_csv("AugSILL_SGDvsDDMD_Jan2021.csv") 

# Get baseline
dmdResults = pd.read_csv("DMD_Feb2021.csv")

# Separate by type
Metrics = list(results.columns)[-7:]
Systems = list(set(results.System))
Algorithms = ['SGD with AugSILL', 'deepDMD']
Kdims = list(set(results.Kdim))
seeds = list(set(results["Random Seed"]))

print("Metrics", Metrics)
print("Systems", Systems)
print("Algorithms", Algorithms)
print("Kdims", Kdims)
print("Random seeds", seeds)

for metric in Metrics:
    fig = plt.figure()
    ax = plt.subplot(111)
    for sys in Systems:
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
            if alg == "SGD with AugSILL":
                epochs = [50*i for i in range(len(means))]
            elif alg == "deepDMD":
                epochs = [250*i for i in range(len(means))]
            else:
                print("WARNING: Unexpected Algorithm!")
                epochs = [i for i in range(len(means))]
            ax.errorbar(epochs, means, yerr=sds, label="sys={0}, \nalg={1}".format(sys, alg))
        plt.legend(framealpha=0)
        if metric == "Test Error5":
            plt.title("Training Epoch vs Log 5-step Prediction Error")
        else:
            plt.title("Training Epoch vs Log {0}".format(metric))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Put in the range of results for standard DMD
        dmd = dmdResults.loc[dmdResults["System"]==sys]
        dmdVals = dmd[metric].to_numpy()
        logDmdVals = np.log(dmdVals)
        dmdMean = np.mean(logDmdVals)
        dmdStd = np.std(logDmdVals)
        ax.plot(epochs, dmdMean * np.ones(len(epochs)), "b", alpha=0.5, label="DMD mean Performance")
        ax.plot(epochs, (dmdMean + dmdStd) * np.ones(len(epochs)), "b--", alpha=0.5)
        ax.plot(epochs, (dmdMean - dmdStd) * np.ones(len(epochs)), "b--", alpha=0.5)
        fig.savefig("Plots/GlycoldDMD_vsSGD_Metric={0}.jpg".format(metric))

