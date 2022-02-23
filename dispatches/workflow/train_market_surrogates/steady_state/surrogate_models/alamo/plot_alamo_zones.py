# produce plot
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', size=40)
plt.rc('axes', titlesize=40)

import pickle
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FuncFormatter

f_inputs = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_dispatch_zones = os.path.join(os.getcwd(),"../../prescient_simulation_sweep_summary_results/prescient_generator_zones.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)

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

with open('models/training_parameters_zones.json', 'r') as outfile:
    data = json.load(outfile)

xm = data['xm_inputs']
xstd = data['xstd_inputs']
zm = data['zm_zones']
zstd = data['zstd_zones']

with open('models/alamo_zone_accuracy.json', 'r') as outfile:
    accuracy_dict = json.load(outfile)

R2 = accuracy_dict['R2']

X_test_scaled = (X_test - xm) / xstd

predictions = []
for zone in zones:
    exec("from models.alamo_zone_{} import f as zone_model".format(zone))
    predicted_hours = np.array([zone_model(*X_test_scaled[i]) for i in range(len(X_test_scaled))])
    predict_unscaled = predicted_hours*zstd[zone] + zm[zone]
    predictions.append(predict_unscaled)


fig, axs = plt.subplots(3,4)
fig.text(0.0, 0.5, 'Predicted Hours in Zone', va='center', rotation='vertical')
fig.text(0.4, 0.02, 'True Hours in Zone', va='center', rotation='horizontal')
fig.set_size_inches(24,24)

titles = ["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
zones_plt = [0,1,2,3,4,5,6,7,8,9,10]
axs_flattened = np.ndarray.flatten(axs)
for (i,zone) in enumerate(zones_plt):
    zt = z_test.transpose()[zone]
    zp = predictions[zone]

    axs_flattened[i].scatter(zt,zp,color = "green",alpha = 0.01)
    axs_flattened[i].plot([min(zt),max(zt)],[min(zt),max(zt)],color = "black")
    if i == 0:
        axs_flattened[i].set_title(titles[zone])
    else:
        axs_flattened[i].set_title(titles[zone] + " of " +"$p_{max}$")

    y_text = 0.75*(max(zt) + min(zt)) - min(zt)
    axs_flattened[i].annotate("$R^2 = {}$".format(round(R2[zone],3)),(0,y_text))


def scientific(x, pos):
    # x:  tick value - ie. what you currently see in yticks
    # pos: a position - ie. the index of the tick (from 0 to 9 in this example)
    return int(x)

scientific_formatter = FuncFormatter(scientific)


for ax in axs_flattened:
    ax.set_aspect('equal')
    #ax.set_xticklabels(ax.get_xticks(), rotation = 25)
    ax.set_yticklabels(ax.get_yticks(), rotation = 45)
    ax.xaxis.set_major_formatter(scientific_formatter)
    ax.yaxis.set_major_formatter(scientific_formatter)
    #ax.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
    #fixed_aspect_ratio(ax,1.0)

axs_flattened[-1].axis('off')

plt.subplots_adjust(wspace=0.3, hspace=0.0, left=0.08, bottom=0.02, right=0.99, top=0.99)
plt.savefig("figures/parity_zone_hours_alamo.png")
plt.savefig("figures/parity_zone_hours_alamo.pdf")
