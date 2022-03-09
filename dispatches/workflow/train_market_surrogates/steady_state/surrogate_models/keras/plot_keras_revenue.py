import tensorflow
from pyomo.common.fileutils import this_file_dir
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from idaes.surrogate.keras_surrogate import KerasSurrogate
from idaes.surrogate.sampling.scaling import OffsetScaler

#make plots
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)

# the data
f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
f_outputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_outputs.h5")
df_inputs = pd.read_hdf(f_inputs)
df_outputs = pd.read_hdf(f_outputs)
predicted_revenue = df_outputs["Total Revenue [$]"]

x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = predicted_revenue.to_numpy()/1e6
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

# the keras model
keras_folder_name = os.path.join(this_file_dir(), 'models','keras_revenue')
keras_model = tensorflow.keras.models.load_model(keras_folder_name, compile=False)

with open('models/training_parameters_revenue.json', 'r') as outfile:
    data = json.load(outfile)

xm = data['xm_inputs']
xstd = data['xstd_inputs']
zm = data['zm_revenue']
zstd = data['zstd_revenue']
input_labels = data['input_labels']
output_labels = data['output_labels']
xmin = data['xmin']
xmax = data['xmax']

inputs_scaler = OffsetScaler(
    expected_columns=input_labels,
    offset_series = pd.Series(dict(zip(input_labels,xm))),
    factor_series = pd.Series(dict(zip(input_labels,xstd))))

outputs_scaler = OffsetScaler(
    expected_columns=output_labels,
    offset_series = pd.Series(dict(zip(output_labels,[zm]))),
    factor_series = pd.Series(dict(zip(output_labels,[zstd]))))

input_bounds={input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))}

keras_surrogate = KerasSurrogate(keras_model=keras_model,
                                 input_labels=input_labels,
                                 output_labels=output_labels,
                                 input_bounds=input_bounds,
                                 input_scaler=inputs_scaler,
                                 output_scaler=outputs_scaler)

X_test_df = pd.DataFrame(X_test,columns=input_labels)
z_predict = keras_surrogate.evaluate_surrogate(X_test_df).to_numpy().flatten()

SS_tot = np.sum(np.square(z_predict - zm))
SS_res = np.sum(np.square(z_test - z_predict))
R2 = round(1 - SS_res/SS_tot,3)

# plot results
plt.figure(figsize=(12,12))
plt.scatter(z_test, z_predict, color = "green", alpha = 0.01)
plt.plot([min(z), max(z)],[min(z), max(z)])
plt.xlabel("True Revenue [MM$]")
plt.ylabel("Predicted Revenue [MM$]")
y_text = 0.75*(max(z) + min(z)) - min(z)
plt.annotate("$R^2 = {}$".format(R2),(0,y_text))
plt.tight_layout()
plt.savefig("figures/revenue_keras.png")
plt.savefig("figures/revenue_keras.pdf")
