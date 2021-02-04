import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


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

results_most = pd.read_csv("AllSysAllDicSGD_Feb2021.csv") 
results_hermite = pd.read_csv("HermiteSGD_Feb2021.csv")
# switch out a column name tht I did wrong.
results_hermite["Algorithm"] = ["SGD with Hermite Polynomials"] * len(results_hermite["Algorithm"])

# Put the results together
results = pd.concat([results_most, results_hermite])

# Separate by type
Metrics = list(results.columns)[-7:]
Systems = list(set(results.System))
Systems.remove("glycolic oscillator") # We don't have this data for most of the algorithms...
Algorithms = list(set(results.Algorithm))
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
            plt.errorbar(epochs, means, yerr=sds, label="sys={0}, alg={1}".format(sys, alg))
        plt.legend()
        plt.title("Training Epoch vs Log {0}\nFor sys={1}".format(metric, sys))
        fig.savefig("Plots/SGD_Metric={0}_sys={1}.jpg".format(metric, sys))

