import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 

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

results = pd.read_csv("AllSysAllDicSGD_Feb2021.csv") 

# Separate by type
Metrics = list(results.columns)[-7:]
Systems = list(set(results.System))
Algorithms = list(set(results.Algorithm))
Kdims = list(set(results.Kdim))
seeds = list(set(results["Random Seed"]))

print("Metrics", Metrics)
print("Systems", Systems)
print("Algorithms", Algorithms)
print("Kdims", Kdims)
print("Random seeds", seeds)

for metric in Metrics:
    fig = plt.figure()
    for sys in Systems:
        for alg in Algorithms:
            # Grab the subset of the data that we want
            # First get the rows
            resPreRow = results.loc[results["System"]==sys]
            resRow = resPreRow.loc[resPreRow["Algorithm"]==alg]
            # Now get the column
            dataStrings = resRow[metric].to_numpy()
            dataBloc = np.array([str2list(stringData) for stringData in dataStrings])
            # Get the mean and standard deviation
            means = np.mean(dataBloc, axis=0)
            sds = np.std(dataBloc, axis=0)
            # Build the plot of score vs error with error bars
            epochs = [50*i for i in range(len(means))]
            plt.errorbar(epochs, means, yerr=sds, label="sys={0}, alg={1}".format(sys, alg))
        plt.legend()
        plt.title("Training Epoch vs {0}".format(metric))
        fig.savefig("Plots/SGD_Metric={0}.jpg".format(metric))

# TODO: When we do this for deepDMD vs SGD we need to account for them not running for the same number of 
# epochs and not having the same samplerate...