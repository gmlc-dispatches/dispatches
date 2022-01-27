from idaes.surrogate import alamopy
import numpy as np
import pandas as pd
import os
import sys
import copy
import pickle
import json
from sklearn.model_selection import  train_test_split

import alamopy_writer

#load prescient data
f_inputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_startups = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_startups.h5")
df_nstartups = pd.read_hdf(f_startups)

x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = df_nstartups["# Startups"].to_numpy()

# 1/3 of data used for test
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

#scale inputs and outputs
xm = np.mean(X_train,axis = 0)
xstd = np.std(X_train,axis = 0)
zm = np.mean(z_train)
zstd = np.std(z_train)

X_train_scaled = (X_train - xm) / xstd
z_train_scaled = (z_train - zm) / zstd

modeler = 1
cases = [[(1,2),(0)],[(1,2),(1)],[(1,2),(1,2)],[(1,2,3),(1,2)],[(1,2,3),(1,2,3)]]
i = 4

result = alamopy.alamo(X_train_scaled,z_train_scaled,
                zlabels = ['nstartups'],
                xlabels=['pmax','pmin_multi','ramp_multi','min_up_time','min_down_multi',
                                'marg_cst','no_load_cst','startup_cst'],
                expandoutput=True,
                showalm=True,
                monomialpower=cases[i][0],
                multi2power=cases[i][1],
                modeler=modeler)

alamopy_writer.write_alamo_func(result['model']['nstartups'],result['xlabels'],'models/alamo_nstartups')
model = result['f(model)']['nstartups']

#remove the lambda function since we can't serialize it with pickle
X_test_scaled = (X_test - xm) / xstd
predicted_revenue = np.array([model(X_test_scaled[i]) for i in range(len(X_test_scaled))])
predict_unscaled = predicted_revenue*zstd + zm

# compute R2 metric
actual_mean = zm
SS_tot = np.sum(np.square(predict_unscaled - actual_mean))
SS_res = np.sum(np.square(z_test - predict_unscaled))
residual = 1 - SS_res/SS_tot
accuracy_dict = {"R2":residual}

result_to_save = copy.copy(result)
result_to_save.pop('f(model)', None)
with open('models/alamo_nstartups.pkl', "wb") as output_file:
    pickle.dump(result_to_save, output_file, pickle.HIGHEST_PROTOCOL)

#save accuracy metrics
with open('models/alamo_nstartups_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

# save scaling and training bounds
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_nstartups":zm,"zstd_nstartups":zstd}

with open('models/training_parameters_nstartups.json', 'w') as outfile:
    json.dump(data, outfile)
