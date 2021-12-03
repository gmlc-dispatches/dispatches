#Use highest accuracy surrogate with all terms
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=18)
plt.rc('axes', titlesize=18)     # fontsize of the axes title
import numpy as np
import pickle
import math

from SALib.sample import saltelli         #saltelli sampler
from SALib.analyze import sobol

f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")
f_perturbed_outputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_outputs.h5")

df_perturbed_inputs = pd.read_csv(f_perturbed_inputs_raw)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

df = df_perturbed_inputs.iloc[:,[1,2,3,4,5,6,7,9]]
cols = df.columns.to_list()
df_bounds = [[min(df.iloc[:,i]),max(df.iloc[:,i])] for i in range(0,8)]


perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]

x = df.to_numpy()
z = perturbed_revenue.to_numpy()/1e6

xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
zm = np.mean(z)
zstd = np.std(z)

x_scaled = (x - xm) / xstd
z_scaled = (z - zm) / zstd

with open("../../surrogate_models/train_surrogates/scikit/models/scikit_revenue.pkl", 'rb') as f:
    model = pickle.load(f)

#SALib problem dict
problem = {'num_vars': 8,'names': cols,'bounds': df_bounds}
param_values = saltelli.sample(problem, 10000, calc_second_order=True)
scaled_param_values = (param_values - xm)/xstd
Y  = np.array([model.predict(inputs.reshape(1,8))[0] for inputs in scaled_param_values])
Y_unscaled = Y*zstd + zm

#Perform Sobol analysis
Si_sobol = sobol.analyze(problem, Y,calc_second_order = True)

S1 = Si_sobol["S1"]
S1_conf = Si_sobol["S1_conf"]
ST = Si_sobol["ST"]
S2 = Si_sobol["S2"]

x = ["X{}".format(i) for i in range(1,9)]
Si_results = {"X":x,"S1":list(S1),"S1_conf":list(S1_conf),"ST":list(ST)}#,"S2":list(S2)}
filename = "salib/Si_sobol_sensitivies.json"
with open(filename, 'w') as f:
    json.dump(Si_results, f)


#Make plots

#First order
x = ["X{}".format(i) for i in range(1,9)]
y = S1
e = Si_sobol["S1_conf"]
fig = plt.figure(figsize = (12,12))
fig.subplots_adjust(left = 0.2)
ax = fig.add_subplot(111)
ax.errorbar(x, y, yerr=e, fmt='o',markersize = 20,capsize = 10)
ax.set_ylabel("Sensitivity Index [Revenue]")
ax.set_xlabel("Input Parameter")
fig.savefig("salib/Si_sobol_1.png")

#Total order
y = ST
e = Si_sobol["ST_conf"]
fig = plt.figure(figsize = (12,12))
fig.subplots_adjust(left = 0.2)
ax = fig.add_subplot(111)
ax.errorbar(x, y, yerr=e, fmt='o',markersize = 20,capsize = 10)
ax.set_ylabel("Sensitivity Index [Revenue]")
ax.set_xlabel("Input Parameter")
fig.savefig("salib/Si_sobol_T.png")

#Second order
matplotlib.rc('font', size=12)
plt.rc('axes', titlesize=12)
x = []
y = []
e = []
for i in range(1,9):
    for j in range(i,9):
        if i != j:
            x.append("X{}X{}".format(i,j))
            y.append(S2[i-1][j-1])
            e.append(Si_sobol["S2_conf"][i-1][j-1])

fig = plt.figure(figsize = (24,8))
fig.subplots_adjust(left = 0.2)
ax = fig.add_subplot(111)
ax.errorbar(x, y, yerr=e, fmt='o',markersize = 20,capsize = 10)
ax.set_ylabel("Sensitivity Index \n [Revenue]")
ax.set_xlabel("Input Parameter")
plt.xticks(rotation = 45)
fig.tight_layout()
fig.savefig("salib/Si_sobol_2.png")
