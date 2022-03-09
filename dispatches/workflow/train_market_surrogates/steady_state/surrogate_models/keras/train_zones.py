# this file trains relu and tanh neural network approximations of a sin-quadratic function
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
tf.keras.backend.set_floatx('float64')

f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_dispatch_zones = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_zones.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)

# scale inputs
x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()

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

model = Sequential(name='revenue')
model.add(Input(8))
model.add(Dense(100, activation='tanh'))
model.add(Dense(100, activation='tanh'))
model.add(Dense(11))
model.compile(optimizer=Adam(), loss='mse')
history = model.fit(x=X_train_scaled, y=z_train_scaled, verbose=1, epochs=500)

model.save('models/keras_zones')

xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":list(xmin),"xmax":list(xmax),
"zm_zones":zm_zones,"zstd_zones":zstd_zones}

input_labels = list(df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].columns)
output_labels = list(df_dispatch_zones.columns[1:])
data["input_labels"] = input_labels
data["output_labels"] = output_labels


with open('models/training_parameters_zones.json', 'w') as outfile:
    json.dump(data, outfile)