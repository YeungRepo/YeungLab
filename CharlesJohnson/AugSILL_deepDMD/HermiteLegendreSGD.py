import numpy as np
from dynamicSystems import *
import pandas as pd
from modelComparisonHelperFunctions import train_test_data_flexDim, train_and_test_SGD



# The full data collection run.
systems = [vdp_system, toggle_system, lv_system, duffing_system, glycol_oscillator]
system_names = ["vdp", "toggle", "lv", "duffing", "glycolic oscillator"]
system_train_lims = [[0, .5], [0, .5], [1, 5], [0, .5], [0, .5]]
system_test_lims = [[0, 1], [0, 1], [1, 9], [0, 1], [0, 1]]
system_dims = [2, 2, 2, 2, 7]
Kdims = [10, 10, 10, 10, 30]

results_df = pd.DataFrame(columns=["System", "Algorithm", "Random Seed", "Kdim", "Time", "Train Error1",
                                    "Train Error2", "Train Error5", "Test Error1", "Test Error2", 
                                    "Test Error5"])

for sys_ind in range(5):
    print("sys_ind", sys_ind)
    for r_seed in [42, 21, 7, 3]:
        print("r_seed", r_seed)
        np.random.seed(r_seed)
        train_data, test_data = train_test_data_flexDim(systems[sys_ind], dim=system_dims[sys_ind], 
                                        nTrain=10, nTest=5,
                                        timeframe=5, 
                                        timesteps=101, 
                                        train_lims=system_train_lims[sys_ind], 
                                        test_lims=system_test_lims[sys_ind])
        
        # SGD with legendre polynomials
        times, tEs1, tEs2, tEs5, TEs1, TEs2, TEs5 = train_and_test_SGD(train_data, test_data, Kdims[sys_ind], dictionary="legendre")
        new_row = {"System": system_names[sys_ind], "Algorithm": "SGD with Legendre Polynomials", "Random Seed": r_seed,
                    "Kdim": Kdims[sys_ind], "Time": times, "Train Error1": tEs1, "Train Error2": tEs2, 
                    "Train Error5": tEs5, "Test Error1": TEs1, "Test Error2": TEs2, 
                    "Test Error5": TEs5}
        results_df = results_df.append(new_row, ignore_index=True)
        print("SDG with legendre done")

        # SGD with hermite polynomials
        times, tEs1, tEs2, tEs5, TEs1, TEs2, TEs5 = train_and_test_SGD(train_data, test_data, Kdims[sys_ind], dictionary="hermite")
        new_row = {"System": system_names[sys_ind], "Algorithm": "SGD with Hermite Polynomials", "Random Seed": r_seed,
                    "Kdim": Kdims[sys_ind], "Time": times, "Train Error1": tEs1, "Train Error2": tEs2, 
                    "Train Error5": tEs5, "Test Error1": TEs1, "Test Error2": TEs2, 
                    "Test Error5": TEs5}
        results_df = results_df.append(new_row, ignore_index=True)
        print("SDG with hermite done")
        
print(results_df)
results_df.to_excel("HermiteLegendreSGD_Feb2021.xlsx") 
results_df.to_csv("HermiteLegendreSGD_Feb2021.csv") 