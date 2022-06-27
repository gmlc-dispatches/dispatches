#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split

#inputs and outputs are in hdf5 files
f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_startups = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_startups.h5")
df_nstartups = pd.read_hdf(f_startups)

x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = df_nstartups["# Startups"].to_numpy()

X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

xm = np.mean(X_train,axis = 0)
xstd = np.std(X_train,axis = 0)
zm = np.mean(z_train)
zstd = np.std(z_train)

X_train_scaled = (X_train - xm) / xstd
z_train_scaled = (z_train - zm) / zstd

# train scikit MLP Regressor model
print("Training NN model ...")
model = MLPRegressor(activation='tanh',hidden_layer_sizes = (100,50)).fit(X_train_scaled, z_train_scaled)

# check the cross validation scores
print("Running Cross Validation Score...")
scores = cross_val_score(model, X_train_scaled, z_train_scaled, cv=5)

# compute model predictions
print("Making NN Predictions...")
X_test_scaled = (X_test - xm) / xstd
predicted_startups = model.predict(X_test_scaled)
predict_unscaled = predicted_startups*zstd + zm

# compute R2 metric
SS_tot = np.sum(np.square(predict_unscaled - zm))
SS_res = np.sum(np.square(z_test - predict_unscaled))
residual = 1 - SS_res/SS_tot
accuracy_dict = {"R2":residual,"CV":list(scores)}

# save model to pickle
print("Saving model ...")
with open("models/scikit_nstartups.pkl", 'wb') as f:
    pickle.dump(model, f)

# save accuracy metrics
with open('models/scikit_nstartups_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

# save scaling and training bounds
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_nstartups":zm,"zstd_nstartups":zstd}

with open('models/training_parameters_nstartups.json', 'w') as outfile:
    json.dump(data, outfile)
