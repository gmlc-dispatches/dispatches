#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

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
from idaes.surrogate.alamopy import AlamoSurrogate

f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)

f_startups = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_startups.h5")
df_nstartups = pd.read_hdf(f_startups)

x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()
z = df_nstartups["# Startups"].to_numpy()
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

with open('models/alamo_parameters_nstartups.json', 'r') as outfile:
    data = json.load(outfile)

xm = data['xm_inputs']
xstd = data['xstd_inputs']
zm = data['zm_nstartups']
zstd = data['zstd_nstartups']

#import the alamo function
alamo_nstartups = AlamoSurrogate.load_from_file(os.path.join('models','alamo_nstartups.json'))
X_test_scaled = (X_test - xm) / xstd
X_test_df = pd.DataFrame(X_test_scaled,columns=xlabels)
zfit = alamo_nstartups.evaluate_surrogate(X_test_df)
predict_unscaled = (zfit*zstd + zm).to_numpy().flatten()

SS_tot = np.sum(np.square(predict_unscaled - zm))
SS_res = np.sum(np.square(z_test - predict_unscaled))
R2 = round(1 - SS_res/SS_tot,3)

# plot results
plt.figure(figsize=(12,12))
plt.scatter(z_test, predict_unscaled, color = "green", alpha = 0.01)
plt.plot([min(z), max(z)],[min(z), max(z)])
plt.xlabel("True # Startups")
plt.ylabel("Predicted # Startups")
y_text = 0.75*(max(z) + min(z)) - min(z)
plt.annotate("$R^2 = {}$".format(R2),(0,y_text))
plt.tight_layout()
plt.savefig("figures/nstartups_alamo.png")
plt.savefig("figures/nstartups_alamo.pdf")