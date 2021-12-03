# This script produces and saves the scikit MLPRegressor surrogate for the revenue data.
# It can use filtered/full or capped/not capped data, and can produce surrogates
# using relu or tanh (see --help for options).

# original script by: S. Martin
# 10/4/2021
from sklearn.neural_network import MLPRegressor
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
import json

#f_perturbed_inputs = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_inputs.h5")
f_perturbed_outputs = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_outputs.h5")

#df_perturbed_inputs = pd.read_hdf(f_perturbed_inputs)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")
df_perturbed_inputs_raw = pd.read_csv(f_perturbed_inputs_raw)

perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]
perturbed_dispatch = df_perturbed_outputs["Total Dispatch [MW]"]

cutoff = 0.0
perturbed_dispatch_array = perturbed_dispatch.to_numpy()
dispatch_inds = np.nonzero(perturbed_dispatch_array >= cutoff*np.max(perturbed_dispatch_array))[0]

# scale revenue data, x is input, z is output
x = df_perturbed_inputs_raw.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()[dispatch_inds]
z = perturbed_revenue.to_numpy()[dispatch_inds]/1e6

xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
zm = np.mean(z)
zstd = np.std(z)

x_scaled = (x - xm) / xstd
z_scaled = (z - zm) / zstd

# train scikit MLP Regressor model
print("Training NN model ...")
model = MLPRegressor(activation='tanh',hidden_layer_sizes = (100,50)).fit(x_scaled, z_scaled)

# compute model predictions
print("Make MLP predictions ...")
predicted_revenue = model.predict(x_scaled)
predict_unscaled = predicted_revenue*zstd + zm

# compute R2 metric
actual_mean = zm
SS_tot = np.sum(np.square(predict_unscaled - actual_mean))
SS_res = np.sum(np.square(z - predict_unscaled))
residual = 1 - SS_res/SS_tot

#Get cross validation score
scores = cross_val_score(model, x_scaled, z_scaled, cv=5)
accuracy_dict = {"R2":residual,"CV":list(scores)}

# save model
print("Saving model ...")
with open("models/scikit_revenue.pkl", 'wb') as f:
    pickle.dump(model, f)

#save accuracy metrics
with open('models/scikit_revenue_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)