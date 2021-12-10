# This script produces and saves the scikit MLPRegressor surrogate for the revenue data.
# It can use filtered/full or capped/not capped data, and can produce surrogates
# using relu or tanh (see --help for options).

# original script by: S. Martin
# 10/4/2021

# sklearn MLPRegressor
from sklearn.neural_network import MLPRegressor
import argparse
import os
import pandas as pd
import numpy as np

# produce plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

# save model
import pickle
import json

f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")
df_perturbed_inputs_raw = pd.read_csv(f_perturbed_inputs_raw)

f_dispatch_zones = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/perturbed_gen_zones_fixed.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)

x = df_perturbed_inputs_raw.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
x_scaled = (x - xm) / xstd

zones = range(0,11)
z_zones_scaled = []
z_zones_unscaled = []
zm_zones = []
zstd_zones = []
for zone in zones:
    z = df_dispatch_zones.iloc[:,zone+1].to_numpy()
    zm = np.mean(z)
    zstd = np.std(z)
    z_scaled = (z - zm) / zstd

    z_zones_unscaled.append(z)
    zm_zones.append(zm)
    zstd_zones.append(zstd)
    z_zones_scaled.append(z_scaled)
    

z_zones_scaled = np.transpose(np.array(z_zones_scaled))
X_train, X_test, z_train, z_test = train_test_split(x_scaled, z_zones_scaled, test_size=0.33, random_state=42)
# train scikit MLP Regressor model
print("Training NN model ...")
model = MLPRegressor(activation='tanh',hidden_layer_sizes = (100,100)).fit(X_train, z_train)
scores = cross_val_score(model, X_train, z_train, cv=5)


# compute model predictions
print("Make MLP predictions ...")
predicted_zones = model.predict(X_test)
predicted_zones_unscaled = predicted_zones*zstd_zones[zone] + zm_zones[zone]
predicted_zones_tp = np.transpose(predicted_zones_unscaled)
z_test_unscaled_tp = np.transpose(z_test*zstd + zm)

R2 = []
for zone in zones:
	SS_tot = np.sum(np.square(predicted_zones_tp[zone] - zm_zones[zone]))
	SS_res = np.sum(np.square(z_test_unscaled_tp[zone] - predicted_zones_tp[zone]))
	residual = 1 - SS_res/SS_tot
	R2.append(residual)

accuracy_dict = {"R2":residual,"CV":list(scores)}

#save model
with open("models/scikit_zones.pkl", 'wb') as f:
    pickle.dump(model, f)


#save accuracy metrics
with open('models/scikit_zone_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_zones":zm_zones,"zstd_zones":zstd_zones}

with open('models/zones_nn_scaling_parameters.json', 'w') as outfile:
    json.dump(data, outfile)