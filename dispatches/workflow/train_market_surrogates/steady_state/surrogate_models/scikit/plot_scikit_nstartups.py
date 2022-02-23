# produce plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)
import os
import pickle
import json
import pandas as pd
from sklearn.model_selection import  train_test_split

f_inputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_startups = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_startups.h5")
df_nstartups = pd.read_hdf(f_startups)

x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = df_nstartups["# Startups"].to_numpy()
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

with open('models/training_parameters_nstartups.json', 'r') as outfile:
    data = json.load(outfile)

xm = data['xm_inputs']
xstd = data['xstd_inputs']
zm = data['zm_nstartups']
zstd = data['zstd_nstartups']

#load up revenue model
with open("models/scikit_nstartups.pkl", 'rb') as f:
    model = pickle.load(f)

with open('models/scikit_nstartups_accuracy.json', 'r') as outfile:
    accuracy_dict = json.load(outfile)

X_test_scaled = (X_test - xm) / xstd
R2 = round(accuracy_dict["R2"],3)
predicted_revenue = model.predict(X_test_scaled)
predict_unscaled = predicted_revenue*zstd + zm

# plot results
plt.figure(figsize=(12,12))
plt.scatter(z_test, predict_unscaled, color = "green", alpha = 0.01)
plt.plot([min(z), max(z)],[min(z), max(z)])
plt.xlabel("True # Startups")
plt.ylabel("Predicted # Startups")
y_text = 0.75*(max(z) + min(z)) - min(z)
plt.annotate("$R^2 = {}$".format(R2),(0,y_text))
plt.tight_layout()
plt.savefig("figures/nstartups_scikit.png")
plt.savefig("figures/nstartups_scikit.pdf")
