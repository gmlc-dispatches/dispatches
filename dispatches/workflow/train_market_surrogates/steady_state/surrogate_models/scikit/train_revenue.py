import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split

#inputs and outputs are in hdf5 files
f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
f_outputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_outputs.h5")
df_inputs = pd.read_hdf(f_inputs)
df_outputs = pd.read_hdf(f_outputs)

predicted_revenue = df_outputs["Total Revenue [$]"]

# inputs --> x, outputs --> z
# column 8 is an integer that maps to a startup profile, whereas column 9 is a representative cost
x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = predicted_revenue.to_numpy()/1e6 #scale dollars to million dollars

# 1/3 of data used for test
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

#scale inputs and outputs
xm = np.mean(X_train,axis = 0)
xstd = np.std(X_train,axis = 0)
zm = np.mean(z_train)
zstd = np.std(z_train)

X_train_scaled = (X_train - xm) / xstd
z_train_scaled = (z_train - zm) / zstd

# train a scikit MLP (multi-layer perceptron) Regressor model
print("Training NN model ...")
#2 layers, 100 nodes, 50 nodes
model = MLPRegressor(activation='tanh',hidden_layer_sizes = (100,50)).fit(X_train_scaled, z_train_scaled)

# these are cross validation scores on training data. normally, this would be done for tuning hyper-parameters.
# this doesn't actually change model parameters, it's just a score.
print("Running Cross Validation Score...")
scores = cross_val_score(model, X_train_scaled, z_train_scaled, cv=5)

# compute model predictions
print("Making NN Predictions...")
X_test_scaled = (X_test - xm) / xstd
predicted_revenue = model.predict(X_test_scaled)
predict_unscaled = predicted_revenue*zstd + zm

# compute R2 metric
SS_tot = np.sum(np.square(predict_unscaled - zm))
SS_res = np.sum(np.square(z_test - predict_unscaled))
residual = 1 - SS_res/SS_tot
accuracy_dict = {"R2":residual,"CV":list(scores)}
print(accuracy_dict)

# save model to pickle
print("Saving model ...")
with open("models/scikit_revenue.pkl", 'wb') as f:
    pickle.dump(model, f)

#save accuracy metrics
with open('models/scikit_revenue_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

#save scaling and training bounds
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_revenue":zm,"zstd_revenue":zstd}

with open('models/training_parameters_revenue.json', 'w') as outfile:
    json.dump(data, outfile)
