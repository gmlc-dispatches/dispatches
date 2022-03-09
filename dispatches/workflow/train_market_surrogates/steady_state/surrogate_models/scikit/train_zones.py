import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, train_test_split

f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_dispatch_zones = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_zones.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)

# scale inputs
x = df_inputs = pd.read_hdf(f_inputs).iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()

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

# train scikit MLP Regressor model, the model has an output for each zone
print("Training NN model ...")
#2 layers, 100 nodes each
model = MLPRegressor(activation='tanh',hidden_layer_sizes = (100,100)).fit(X_train_scaled, z_train_scaled)
scores = cross_val_score(model, X_train_scaled, z_train_scaled, cv=5)

print("Make predictions ...")
predicted_hours = np.array(model.predict(X_test_scaled))
predict_unscaled = predicted_hours*zstd + zm

# compute model predictions
test_R2 = []
for zone in range(0,11):
    # compute R2 metric
    zpredict = predict_unscaled.transpose()[zone]
    SS_tot = np.sum(np.square(zpredict - zm[zone]))
    SS_res = np.sum(np.square(z_test.transpose()[zone] - zpredict))
    residual = 1 - SS_res/SS_tot
    test_R2.append(residual)

accuracy_dict = {"R2":test_R2,"CV":list(scores)}

# save model
with open("models/scikit_zones.pkl", 'wb') as f:
    pickle.dump(model, f)

# save accuracy metrics
with open('models/scikit_zone_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

# save training bounds and scaling
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_zones":zm_zones,"zstd_zones":zstd_zones}

with open('models/training_parameters_zones.json', 'w') as outfile:
    json.dump(data, outfile)
