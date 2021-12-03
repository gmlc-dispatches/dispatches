#Global sensitivity analysis on original Prescient data set
import pandas as pd
import os
import json
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=18)
plt.rc('axes', titlesize=18)     # fontsize of the axes title
import numpy as np
from SALib.analyze import delta,rbd_fast, morris

f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")
f_perturbed_outputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_outputs.h5")

df_perturbed_inputs_raw = pd.read_csv(f_perturbed_inputs_raw)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]
perturbed_dispatch = df_perturbed_outputs["Total Dispatch [MW]"]

#we do not cutoff data anymore
cutoff = 0.0
perturbed_dispatch_array = perturbed_dispatch.to_numpy()
dispatch_inds = np.nonzero(perturbed_dispatch_array >= cutoff*np.max(perturbed_dispatch_array))[0]
df_perturbed_inputs_cutoff = df_perturbed_inputs_raw.iloc[dispatch_inds,1:9]
perturbed_revenue_cutoff = perturbed_revenue.to_numpy()[dispatch_inds]

#ANALYSIS ON TRUE DATASET
df = df_perturbed_inputs_cutoff
cols = df.columns.to_list()
df_bounds = [[min(df.iloc[:,i]),max(df.iloc[:,i])] for i in range(0,8)]

#SALib problem dict
problem = {'num_vars': 8,'names': cols,'bounds': df_bounds}
X = df.to_numpy()
Y = perturbed_revenue_cutoff/1e6

#Perform Delta Moment-Independent Analysis
Si_delta = delta.analyze(problem, X, Y, print_to_console=True)
x = ["X{}".format(i) for i in range(1,9)]
y_delta = Si_delta["S1"]
e = Si_delta["S1_conf"]

Si_results = {"X":x,"S1":list(y_delta),"S1_conf":list(e)}
filename = "salib/Si_delta_sensitivies.json"
with open(filename, 'w') as f:
    json.dump(Si_results, f)

fig = plt.figure(figsize = (12,12))
fig.subplots_adjust(left = 0.2)
ax = fig.add_subplot(111)
ax.errorbar(x, y_delta, yerr=e, fmt='o',markersize = 20,capsize = 10)
ax.set_ylabel("Sensitivity Index [Revenue]")
ax.set_xlabel("Input Parameter")
fig.savefig("salib/Si_delta.png")
fig.savefig("salib/Si_delta.pdf")

#Perform Random Balanced Design - Fourier Amplitude Sensitivity Test
Si_rbdfast = rbd_fast.analyze(problem,X,Y)
y = Si_rbdfast["S1"]
fig = plt.figure(figsize = (12,12))
fig.subplots_adjust(left = 0.2)
ax = fig.add_subplot(111)
ax.scatter(x,y,s = 400)
ax.set_ylabel("Sensitivity Index [Revenue]")
ax.set_xlabel("Input Parameter")
# fig.tight_layout()
fig.savefig("salib/Si_rbdfast.png")
fig.savefig("salib/Si_rbdfast.pdf")
#Si sums to 1.0!
