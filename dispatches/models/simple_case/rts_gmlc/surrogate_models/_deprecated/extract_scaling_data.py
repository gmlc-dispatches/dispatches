#This script extracts the scaling data and input limits used to train the neural network surrogates and saves it in a json file
#TODO: finish this script. right now all the scaling is pulled from the training files
import os
import pandas as pd
import numpy as np
import pickle
import json

f_perturbed_inputs_raw = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_input_combinations.csv")
df_perturbed_inputs = pd.read_csv(f_perturbed_inputs_raw)

f_perturbed_outputs = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/prescient_perturbed_gen_outputs.h5")
df_perturbed_outputs = pd.read_hdf(f_perturbed_outputs)

f_dispatch_zones = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/perturbed_gen_zones_fixed.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)

f_nstartups = os.path.join(os.getcwd(),"../../../prescient_simulation_sweep_summary_results/perturbed_gen_startups.h5")
df_nstartups = pd.read_hdf(f_nstartups)

dispatch_zones = df_dispatch_zones.columns[1:]
perturbed_revenue = df_perturbed_outputs["Total Revenue [$]"]

# scale revenue data, x is input, z is output
x = df_perturbed_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()

#input scaling
xm = np.mean(x,axis = 0)
xstd = np.std(x,axis = 0)
#input bounds 
xmin = list(np.min(x,axis=0))
xmax = list(np.max(x,axis=0))

z_revenue = perturbed_revenue.to_numpy()/1e6

zm_rev = np.mean(z)
zstd_rev = np.std(z)

z_start = df_nstartups["# Startups"].to_numpy()

zm_start = np.mean(z)
zstd_start = np.std(z)


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