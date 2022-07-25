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
matplotlib.rc('font', size=40)
plt.rc('axes', titlesize=40)

import json
import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans,silhouette_score
import pickle
from tslearn_test_6400_years import TSA64K
from sklearn.model_selection import train_test_split
from TSA_NN_surrogate import read_input_x, calculate_ws, load_cluster_model

input_file = 'prescient_generator_inputs.h5'
dispatch_csv = 'Dispatch_shuffled_data_0.csv'
mdclustering = 'result_6400years_shuffled_30clusters_OD.json'

# read the ws (years, 32)
clustering_model = load_cluster_model(mdclustering)
ws = calculate_ws(clustering_model, dispatch_csv)

# refer to Jordan's codes 
x = read_input_x(input_file, dispatch_csv)

# load the NN model
with open('NN_model_params/scikit_NN_ws.pkl', 'rb') as f:
    NN_model = pickle.load(f)

# load accuracy data
# with open('NN_model_params/scikit_NN_ws_accuracy.json', 'r') as f1:
#     accuracy_dict = json.load(f1)

# load scaling bounds
with open('NN_model_params/training_params_ws.json', 'r') as f2:
	model_params = json.load(f2)

# split the train/test data
x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=42)

# scaling bounds
xm = model_params['xm_inputs']
xstd = model_params['xstd_inputs']
wsm = model_params['ws_mean']
wsstd = model_params['ws_std']

x_test_scaled = (x_test - xm)/xstd
pred_ws = NN_model.predict(x_test_scaled)
# np.shape(pred_ws_unscaled) = (years, 32)
pred_ws_unscaled = pred_ws*wsstd + wsm

# accuracy params
R2 = []
for rd in range(0,32):
    # compute R2 metric
    wspredict = pred_ws_unscaled.transpose()[rd]
    SS_tot = np.sum(np.square(wspredict - wsm[rd]))
    SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
    residual = 1 - SS_res/SS_tot
    R2.append(residual)

# sub plots 4*4 in one figure
fig1, axs1 = plt.subplots(4,4)
fig1.text(0.0, 0.5, 'Predicted demand frequency', va='center', rotation='vertical')
fig1.text(0.5, 0.03, 'True demand frequency', va='center', rotation='horizontal')
fig1.set_size_inches(48,48)

titles = []
for i in range(32):
	titles.append(f'cluster_{i}')

plt_num_1 = list(range(16))
plt_num_2 = list(range(16,32))
axs1_flattened = np.ndarray.flatten(axs1)

for idx, w in enumerate(plt_num_1):
    wst = ws_test.transpose()[w]
    wsp = pred_ws_unscaled.transpose()[w]

    axs1_flattened[idx].scatter(wst,wsp,color = "green",alpha = 0.05)
    axs1_flattened[idx].plot([min(wst),max(wst)],[min(wst),max(wst)],color = "black")
    axs1_flattened[idx].set_title(titles[w])

    axs1_flattened[idx].annotate("$R^2 = {}$".format(round(R2[w],3)),(0,0.75*max(wst)))

plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.08, bottom=0.05, right=0.99, top=0.97)
plt.savefig("figures/plot_ws_scikit_0.png")

fig2, axs2 = plt.subplots(4,4)
fig2.text(0.0, 0.5, 'Predicted demand frequency', va='center', rotation='vertical')
fig2.text(0.5, 0.03, 'True demand frequency', va='center', rotation='horizontal')
fig2.set_size_inches(48,48)

axs2_flattened = np.ndarray.flatten(axs2)

for idx, w in enumerate(plt_num_2):
    wst = ws_test.transpose()[w]
    wsp = pred_ws_unscaled.transpose()[w]

    axs2_flattened[idx].scatter(wst,wsp,color = "green",alpha = 0.05)
    axs2_flattened[idx].plot([min(wst),max(wst)],[min(wst),max(wst)],color = "black")
    axs2_flattened[idx].set_title(titles[w])

    axs2_flattened[idx].annotate("$R^2 = {}$".format(round(R2[w],3)),(0,0.75*max(wst)))

plt.subplots_adjust(wspace=0.3, hspace=0.15, left=0.08, bottom=0.05, right=0.99, top=0.97)
plt.savefig("figures/plot_ws_scikit_1.png")
