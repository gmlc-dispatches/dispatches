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

from idaes.surrogate import alamopy
import numpy as np
import pandas as pd
import os, sys, copy, pickle, json
from sklearn.model_selection import  train_test_split
from idaes.surrogate.alamopy import AlamoTrainer, AlamoSurrogate

#load prescient data
f_inputs = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_inputs.h5")
df_inputs = pd.read_hdf(f_inputs)
f_dispatch_zones = os.path.join(os.getcwd(),"../../prescient_data/prescient_generator_zones.h5")
df_dispatch_zones = pd.read_hdf(f_dispatch_zones)
x = df_inputs.iloc[:,[1,2,3,4,5,6,7,9]].to_numpy()

# zone outputs contain number of hours the generator operates within a range of power output
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
xm = np.mean(X_train, axis=0)
xstd = np.std(X_train, axis=0)
zm = np.mean(z_train, axis=0)
zstd = np.std(z_train, axis=0)

X_train_scaled = (X_train - xm) / xstd
z_train_scaled = (z_train - zm) / zstd
X_test_scaled = (X_test - xm) / xstd


# IDAES.Surrogates
numins = np.shape(X_train_scaled)[1]
numouts = np.shape(z_train_scaled)[1]

xmin = list(X_train_scaled.min(axis=0))
xmax = list(X_train_scaled.max(axis=0))
xlabels=['pmax','pmin_multi','ramp_multi','min_up_time','min_down_multi',
         'marg_cst','no_load_cst','startup_cst']
zlabels = ['zone_{}'.format(zone) for zone in range(0,11)]
labels = xlabels + zlabels

data_in = pd.DataFrame(X_train_scaled,columns=xlabels)
data_out = pd.DataFrame(z_train_scaled,columns=zlabels)
data = pd.concat([data_in,data_out],axis=1)
input_bounds = {xlabels[i]: (xmin[i], xmax[i]) for i in range(numins)}

trainer = AlamoTrainer(input_labels=xlabels,
                       output_labels=zlabels,
                       input_bounds=input_bounds,
                       training_dataframe=data)
trainer._n_inputs = numins
trainer._n_outputs = numouts
trainer._rdata_in = data_in
trainer._rdata_out = data_out

#set options
aoptlabels = ['constant', 'linfcns', 'multi2power', 'monomialpower',
              'maxterms', 'filename', 'overwrite_files','maxtime']
filename = os.path.join(os.getcwd(), 'alamo_run.alm')
aoptvals = [True, True, (1, 2, 3), (2, 3), [15] * len(zlabels),
            filename, True, 3600]
alamopy_options = dict(zip(aoptlabels, aoptvals))
for entry in alamopy_options:
    setattr(trainer.config, entry, alamopy_options[entry])

# train alamo surrogate and save
success, alm_surr, msg = trainer.train_surrogate()
alm_surr.save_to_file('models/alamo_zones.json', overwrite=True)

X_test_scaled = (X_test - xm) / xstd
X_test_df = pd.DataFrame(X_test_scaled,columns=xlabels)
zfit = alm_surr.evaluate_surrogate(X_test_df)
predict_unscaled = (zfit*zstd + zm)#.to_numpy()#.flatten()

# compute R2 metric and save
SS_tot = np.sum(np.square(predict_unscaled - zm),axis=0)
SS_res = np.sum(np.square(z_test - predict_unscaled),axis=0)
residual = 1 - SS_res/SS_tot
accuracy_dict = {"R2":list(residual)}
with open('models/alamo_zone_accuracy.json', 'w') as outfile:
    json.dump(accuracy_dict, outfile)

# save scaling and training bounds
xmin = list(np.min(X_train,axis=0))
xmax = list(np.max(X_train,axis=0))
data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
"zm_zones":list(zm),"zstd_zones":list(zstd)}

with open('models/alamo_parameters_zones.json', 'w') as outfile:
    json.dump(data, outfile)
