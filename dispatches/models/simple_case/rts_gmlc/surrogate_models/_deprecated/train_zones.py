from idaes.surrogate import alamopy
import numpy as np
import pandas as pd
import os
import sys
import copy
import pickle
import json
from sklearn.model_selection import train_test_split

import alamopy_writer

f_inputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_dispatch_zones = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_zones.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)
x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()

# zone outputs contain number of hours the generator operates within a range of power output
zones = range(0,11)
z_zones_unscaled = []
zm_zones = []
zstd_zones = []
for zone in zones:
    z = df_dispatch_zones.iloc[:,zone+1].to_numpy()
    zm = np.mean(z)
    zstd = np.std(z)
    zm_zones.append(zm)
    zstd_zones.append(zstd)
    z_zones_unscaled.append(z)

X_train, X_test, z_train, z_test = train_test_split(x, np.transpose(z_zones_unscaled), test_size=0.33, random_state=42)
xm = np.mean(X_train, axis=0)
xstd = np.std(X_train, axis=0)
zm = np.mean(z_train, axis=0)
zstd = np.std(z_train, axis=0)

X_train_scaled = (X_train - xm) / xstd
z_train_scaled = (z_train - zm) / zstd
X_test_scaled = (X_test - xm) / xstd

modeler = 1
cases = [[(1,2),(0)],[(1,2),(1)],[(1,2),(1,2)],[(1,2,3),(1,2)],[(1,2,3),(1,2,3)]]
i = 4
test_R2 = []
for zone in range(0,11):
    zdata = z_train_scaled.transpose()[zone]
    result = alamopy.alamo(X_train_scaled,zdata,
                    zlabels = ['zone_{}'.format(zone)],
                    xlabels=['pmax','pmin_multi','ramp_multi','min_up_time','min_down_multi',
                                    'marg_cst','no_load_cst','startup_cst'],
                    expandoutput=True,
                    showalm=True,
                    monomialpower=cases[i][0],
                    multi2power=cases[i][1],
                    modeler=modeler)

    alamopy_writer.write_alamo_func(result['model']['zone_{}'.format(zone)],result['xlabels'],'models/alamo_zone_{}'.format(zone))
    model = result['f(model)']['zone_{}'.format(zone)]

    predicted_hours = np.array([model(X_test_scaled[i]) for i in range(len(X_test_scaled))])
    predict_unscaled = predicted_hours*zstd[zone] + zm[zone]

    # compute R2 metric
    SS_tot = np.sum(np.square(predict_unscaled - zm[zone]))
    SS_res = np.sum(np.square(z_test.transpose()[zone] - predict_unscaled))
    residual = 1 - SS_res/SS_tot
    test_R2.append(residual)

    result_to_save = copy.copy(result)
    result_to_save.pop('f(model)', None)
    with open('models/alamo_zone_{}.pkl'.format(zone), "wb") as output_file:
        pickle.dump(result_to_save, output_file, pickle.HIGHEST_PROTOCOL)

accuracy_dict = {"R2":test_R2}

#save accuracy metrics
with open('models/alamo_zone_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

# save scaling and training bounds
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_zones":list(zm),"zstd_zones":list(zstd)}

with open('models/training_parameters_zones.json', 'w') as outfile:
    json.dump(data, outfile)