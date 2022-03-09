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

#train tanh on scaled data
model = Sequential(name='revenue')
model.add(Input(8))
model.add(Dense(100, activation='tanh'))
model.add(Dense(50, activation='tanh'))
model.add(Dense(1))
model.compile(optimizer=Adam(), loss='mse')
history = model.fit(x=X_train_scaled, y=z_train_scaled, verbose=1, epochs=500)

model.save('models/keras_revenue')

xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_revenue":zm,"zstd_revenue":zstd}

input_labels = list(df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].columns)
data["input_labels"] = input_labels
data["output_labels"] = ["Revenue [MM$]"]

with open('models/training_parameters_revenue.json', 'w') as outfile:
    json.dump(data, outfile)

