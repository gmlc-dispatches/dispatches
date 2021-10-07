#This script extracts the scaling used to train the neural network surrogates and saves it in a json file
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import json

# load Prescient revenue dataset
f_perturbed_inputs = os.path.join(os.getcwd(),"../run_prescient/simulation_sweep_summary/prescient_perturbed_gen_inputs.h5")
f_perturbed_outputs = os.path.join(os.getcwd(),"../run_prescient/simulation_sweep_summary/prescient_perturbed_gen_outputs.h5")
f_dispatch_zones = os.path.join(os.getcwd(),"../run_prescient/simulation_sweep_summary/perturbed_gen_zones.h5")

# read data
df_perturbed_inputs = pd.read_hdf(f_perturbed_inputs)
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)

# identify revenue/dispatch
dispatch_zones = df_dispatch_zones.columns[1:]
perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]
perturbed_dispatch = df_perturbed_outputs["Total Dispatch [MW]"]
perturbed_dispatch_array = perturbed_dispatch.to_numpy()

#cutoff dispatch
cutoff = 0.0
dispatch_inds = np.nonzero(perturbed_dispatch_array >= cutoff*np.max(perturbed_dispatch_array))[0]


# input scaling
x = df_perturbed_inputs.iloc[:,1:].to_numpy()[dispatch_inds]
xm = list(np.mean(x,axis = 0))
xstd = list(np.std(x,axis = 0))

#input bounds 
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))

# revenue scaling
z = perturbed_revenue.to_numpy()[dispatch_inds]/1e6
zm_revenue = np.mean(z)
zstd_revenue = np.std(z)


#zone scaling
zones = range(0,11)
zone_offsets = []
zone_factors = []
for zone in zones:

    # hour in dispatch zone
    output_array = df_dispatch_zones.iloc[:,zone+1].to_numpy()
    z = output_array[dispatch_inds]

    # mean center, unit scaling
    zm = np.mean(z)
    zstd = np.std(z)

    zone_offsets.append(zm)
    zone_factors.append(zstd)

data = {
"xm_inputs":xm,"xstd_inputs":xstd,"xmin":xmin,"xmax":xmax,
"zm_revenue":zm_revenue,"zstd_revenue":zstd_revenue,
"zm_zones":zone_offsets,"zstd_zones":zone_factors}

with open('prescient_scaling_parameters.json', 'w') as outfile:
    json.dump(data, outfile)