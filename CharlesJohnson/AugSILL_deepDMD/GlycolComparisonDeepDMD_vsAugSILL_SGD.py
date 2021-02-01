import numpy as np
from dynamicSystems import glycol_oscillator
import pandas as pd
from modelComparisonHelperFunctions import train_test_data_flexDim, train_and_test_SGD, train_and_test_FFdeepDMD


system_name = "glycolic oscillator"
system_train_lims = [0, .5]
system_test_lims = [0, 1]

results_df = pd.DataFrame(columns=["System", "Algorithm", "Random Seed", "Kdim", "Time", "Train Error1",
                                    "Train Error2", "Train Error5", "Test Error1", "Test Error2", 
                                    "Test Error5"])

# we add 20 dictionary elements for deepDMD and we add a bias term and 19 dictionary elements for AugSILL.
Kdim = 27 
for r_seed in [42, 21, 7, 3]:
    print("r_seed", r_seed)
    np.random.seed(r_seed)
    train_data, test_data = train_test_data_flexDim(glycol_oscillator, dim=7, nTrain=10, nTest=5,
                                            timeframe=5, 
                                            timesteps=101, 
                                            train_lims=system_train_lims, 
                                            test_lims=system_test_lims)
    # SGD with AugSILL
    times, tEs1, tEs2, tEs5, TEs1, TEs2, TEs5 = train_and_test_SGD(train_data, test_data, Kdim)
    new_row = {"System": system_name, "Algorithm": "SGD with AugSILL", "Random Seed": r_seed,
                "Kdim": Kdim, "Time": times, "Train Error1": tEs1, "Train Error2": tEs2, 
                "Train Error5": tEs5, "Test Error1": TEs1, "Test Error2": TEs2, 
                "Test Error5": TEs5}
    results_df = results_df.append(new_row, ignore_index=True)
    print("SDG with AugSILL done")
    # DeepDMD with Feedforward Layers
    times, tEs1, tEs2, tEs5, TEs1, TEs2, TEs5 = train_and_test_FFdeepDMD(train_data, test_data, Kdim)
    new_row = {"System": system_name, "Algorithm": "deepDMD", "Random Seed": r_seed,
                "Kdim": Kdim, "Time": times, "Train Error1": tEs1, "Train Error2": tEs2, 
                "Train Error5": tEs5, "Test Error1": TEs1, "Test Error2": TEs2, 
                "Test Error5": TEs5}
    results_df = results_df.append(new_row, ignore_index=True)
    print("deepDMD done")

print(results_df)
results_df.to_excel("AugSILL_SGDvsDDMD_Jan2021.xlsx") 
results_df.to_csv("AugSILL_SGDvsDDMD_Jan2021.csv") 