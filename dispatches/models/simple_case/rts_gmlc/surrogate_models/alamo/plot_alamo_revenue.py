# produce plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)
import pickle
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split

f_inputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_inputs.h5")
f_outputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_outputs.h5")
df_inputs = pd.read_hdf(f_inputs)
df_outputs = pd.read_hdf(f_outputs)
predicted_revenue = df_outputs["Total Revenue [$]"]

x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = predicted_revenue.to_numpy()/1e6
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

with open('models/training_parameters_revenue.json', 'r') as outfile:
    data = json.load(outfile)

xm = data['xm_inputs']
xstd = data['xstd_inputs']
zm = data['zm_revenue']
zstd = data['zstd_revenue']

#import the alamo function
exec("from models.alamo_revenue import f as model")
X_test_scaled = (X_test - xm) / xstd
predicted_revenue = np.array([model(*X_test_scaled[i]) for i in range(len(X_test_scaled))])
predict_unscaled = predicted_revenue*zstd + zm

with open('models/alamo_revenue_accuracy.json', 'r') as outfile:
    accuracy_dict = json.load(outfile)
R2 = round(accuracy_dict["R2"],3)

# plot results
plt.figure(figsize=(12,12))
plt.scatter(z_test, predict_unscaled, color = "green", alpha = 0.01)
plt.plot([min(z), max(z)],[min(z), max(z)])
plt.xlabel("True Revenue [MM$]")
plt.ylabel("Predicted Revenue [MM$]")
y_text = 0.75*(max(z) + min(z)) - min(z)
plt.annotate("$R^2 = {}$".format(R2),(0,y_text))
plt.tight_layout()
plt.savefig("figures/revenue_alamo.png")
plt.savefig("figures/revenue_alamo.pdf")