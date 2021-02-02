import numpy as np
from dynamicSystems import vdp_system, toggle_system, lv_system, duffing_system
import pandas as pd
from modelComparisonHelperFunctions import train_test_data, train_and_test_SGD_old, train_and_test_MP_full, train_and_test_MP_simple


def get_center_vals(min_val, max_val, num=8):
    centers = []
    vals = np.linspace(min_val, max_val, num)
    for i in range(num):
        for j in range(num):
            centers.append([vals[i], vals[j]])
    return centers

# matching pursuit preparation.
steepness_vals = np.linspace(1, 10, 4)
steepnesses = []
for i in range(len(steepness_vals)):
    for j in range(len(steepness_vals)):
        steepnesses.append([steepness_vals[i], steepness_vals[j]])
    
vdp_centers = get_center_vals(-.75, .75)
togg_centers = get_center_vals(-.15, .75)
lv_centers = get_center_vals(0, 5)
duff_centers = get_center_vals(-.75, .75)

# The full data collection run.
systems = [vdp_system, toggle_system, lv_system, duffing_system]
system_names = ["vdp", "toggle", "lv", "duffing"]
system_train_lims = [[0, .5], [0, .5], [1, 5], [0, .5]]
system_test_lims = [[0, 1], [0, 1], [1, 9], [0, 1]]
system_centers = [vdp_centers, togg_centers, lv_centers, duff_centers]


results_df = pd.DataFrame(columns=["System", "Algorithm", "Random Seed", "Kdim", "Time", "Train Error1", "Train Error2", "Train Error5", 
                                   "Test Error1", "Test Error2", "Test Error5", "Num Basis Elements"])
for sys_ind in range(4): 
    print("sys_ind", sys_ind)
    for Kdim in [5, 10, 20]:
        print("Kdim", Kdim)
        for r_seed in [42, 21, 7, 3]:
            print("r_seed", r_seed)
            np.random.seed(r_seed)
            train_data, test_data = train_test_data(systems[sys_ind], timeframe=15, timesteps=16, 
                                                    train_lims=system_train_lims[sys_ind], 
                                                    test_lims=system_test_lims[sys_ind])
            tyme, train_error1, train_error2, train_error5, test_error1, test_error2, test_error5, logC, logS, rbfC, rbfS = train_and_test_SGD_old(
                train_data, test_data, Kdim)
            train_error1 = train_error1.item()
            train_error2 = train_error2.item()
            train_error5 = train_error5.item()
            test_error1 = test_error1.item()
            test_error2 = test_error2.item()
            test_error5 = test_error5.item()
            num_basis = -1
            new_row = {"System": system_names[sys_ind], "Algorithm": "SGD", "Random Seed": r_seed,
                       "Kdim": Kdim, "Time": tyme, "Train Error1": train_error1, "Train Error2": train_error2, 
                       "Train Error5": train_error5, "Test Error1": test_error1, "Test Error2": test_error2, 
                       "Test Error5": test_error5, "Num Basis Elements": num_basis}
            results_df = results_df.append(new_row, ignore_index=True)
            print("SDG done")
            tyme, train_error1, train_error2, train_error5, test_error1, test_error2, test_error5, num_basis = train_and_test_MP_simple(train_data, test_data, 
                                                                                steepnesses, 
                                                                                system_centers[sys_ind], 
                                                                                Kdim)
            new_row = {"System": system_names[sys_ind], "Algorithm": "Simple Matching Pursuit", "Random Seed": r_seed,
                       "Kdim": Kdim, "Time": tyme, "Train Error1": train_error1, "Train Error2": train_error2, 
                       "Train Error5": train_error5, "Test Error1": test_error1, "Test Error2": test_error2, 
                       "Test Error5": test_error5, "Num Basis Elements": num_basis}
            results_df = results_df.append(new_row, ignore_index=True)
            print("Simple MP done")
            
            tyme, train_error1, train_error2, train_error5, test_error1, test_error2, test_error5, num_basis = train_and_test_MP_full(train_data, test_data, 
                                                                              steepnesses, 
                                                                              system_centers[sys_ind], Kdim)
            new_row = {"System": system_names[sys_ind], "Algorithm": "Full Matching Pursuit", "Random Seed": r_seed,
                       "Kdim": Kdim, "Time": tyme, "Train Error1": train_error1, "Train Error2": train_error2, 
                       "Train Error5": train_error5, "Test Error1": test_error1, "Test Error2": test_error2, 
                       "Test Error5": test_error5, "Num Basis Elements": num_basis}
            results_df = results_df.append(new_row, ignore_index=True)
            print("Full MP done")
            
            
print(results_df)
results_df.to_excel("SGDvsMP_Dec2020.xlsx") 
results_df.to_csv("SGDvsMP_Dec2020.csv") 