# produce plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=18)
plt.rc('axes', titlesize=18)

import pickle
import os
import pandas as pd
import numpy as np

f_perturbed_outputs = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_outputs.h5")
f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")

df_perturbed_inputs_raw = pd.read_csv(f_perturbed_inputs_raw)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

f_dispatch_zones = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/perturbed_gen_zones_fixed.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)


# scale revenue data, x is input, z is output
x = df_perturbed_inputs_raw.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()

xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
x_scaled = (x - xm) / xstd

zones = range(0,11)
z_zones_scaled = []
z_zones_unscaled = []
zm_zones = []
zstd_zones = []
for zone in zones:
    z = df_dispatch_zones.iloc[:,zone+1].to_numpy()
    zm = np.mean(z)
    zstd = np.std(z)
    z_scaled = (z - zm) / zstd

    z_zones_unscaled.append(z)
    zm_zones.append(zm)
    zstd_zones.append(zstd)
    z_zones_scaled.append(z_scaled)

#load up revenue model
with open("models/scikit_zones.pkl", 'rb') as f:
    model = pickle.load(f)
predicted_zones = model.predict(x_scaled)
predicted_zones_tp = np.transpose(predicted_zones)

z_predict_unscaled = []
for zone in zones:
	z_zone_scaled = predicted_zones_tp[zone]
	predict_unscaled = z_zone_scaled*zstd_zones[zone] + zm_zones[zone]
	z_predict_unscaled.append(predict_unscaled)


#import matplotlib.gridspec as gridspec
fig, axs = plt.subplots(2,6)
fig.text(0.0, 0.5, 'Predicted Hours in Zone', va='center', rotation='vertical')
fig.text(0.4, 0.02, 'True Hours in Zone', va='center', rotation='horizontal')
fig.set_size_inches(18,6)


titles = ["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
zones_plt = [0,1,2,3,4,5,6,7,8,9,10]
axs_flattened = np.ndarray.flatten(axs)
for (i,zone) in enumerate(zones_plt):
    z = z_zone_unscaled[zone]
    unscaled_predicted_hours = z_predict_unscaled[zone]

    axs_flattened[i].scatter(z,unscaled_predicted_hours,color = "green",alpha = 0.01)
    axs_flattened[i].plot([min(z),max(z)],[min(z),max(z)],color = "black")
    if i == 0:
        axs_flattened[i].set_title(titles[zone])
    else:
        axs_flattened[i].set_title(titles[zone] + " of " +"$p_{max}$")

for ax in axs_flattened:
	ax.set_aspect('equal')
    #fixed_aspect_ratio(ax,1.0)

axs_flattened[-1].axis('off')

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.tight_layout()
plt.savefig("figures/parity_zone_hours.png")

matplotlib.rc('font', size=32)
plt.rc('axes', titlesize=32)
plt.cla()
fig = plt.figure()
fig.set_size_inches(12,12)
zone = 0
z = z_zones_unscaled[zone]
unscaled_predicted_hours = z_predict_unscaled[zone]
plt.scatter(z,unscaled_predicted_hours,color = "green",alpha = 0.01)
plt.plot([min(z),max(z)],[min(z),max(z)],color = "black")
plt.xlabel("True Hours Shutdown")
plt.ylabel("Predicted Hours Shutdown")
plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.tight_layout()
plt.savefig("figures/parity_zone_hours_0.png")
