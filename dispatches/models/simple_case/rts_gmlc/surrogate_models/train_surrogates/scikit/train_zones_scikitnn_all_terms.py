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

# train scikit MLP Regressor model
print("Training NN model ...")
model = MLPRegressor(activation='tanh',hidden_layer_sizes = (100,100)).fit(x_scaled, z_zones_scaled)

scores = cross_val_score(model, x_scaled, z_zones_scaled, cv=5)


# compute model predictions
print("Make MLP predictions ...")
predicted_zones = model.predict(x_scaled)
predicted_zones_tp = np.transpose(predicted_zones)

z_predict_unscaled = []
R2 = []
for zone in zones:
	z_zone_scaled = predicted_zones_tp[zone]
	predict_unscaled = z_zone_scaled*zstd_zones[zone] + zm_zones[zone]
	z_predict_unscaled.append(predict_unscaled)

	SS_tot = np.sum(np.square(predict_unscaled - zm_zones[zone]))
	SS_res = np.sum(np.square(z_zones_unscaled[zone] - predict_unscaled))
	residual = 1 - SS_res/SS_tot
	R2.append(residual)

accuracy_dict = {"R2":residual,"CV":list(scores)}

#save model
with open("models/scikit_zones.pkl", 'wb') as f:
    pickle.dump(model, f)


#save accuracy metrics
with open('models/scikit_zone_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)