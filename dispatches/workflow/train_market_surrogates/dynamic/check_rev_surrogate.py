import os

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from idaes.core.util import to_json, from_json
import numpy as np
import json
import re
import matplotlib.pyplot as plt
from Simulation_Data import SimulationData
from Train_NN_Surrogates import TrainNNSurrogates
from Time_Series_Clustering import TimeSeriesClustering
import pandas as pd

# for NE case study
# case_name = 'NE'
# dispatch_data_path = '../../../../../datasets/results_nuclear_sweep/Dispatch_data_NE_whole.csv'
# input_data_path = '../../../../../datasets/results_nuclear_sweep/sweep_parameters_results_NE_whole.h5'
# rev_data_path = '../../../../../datasets/results_nuclear_sweep/NE_revenue.csv'
# num_clusters = 20
# num_sims = 192
# input_layer_node = 4
# filter_opt = True

# for RE case study
# case_name = 'RE'
# dispatch_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_whole.csv'
# input_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
# rev_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2/RE_H2_revenue.csv'
# num_clusters = 20
# num_sims = 224
# input_layer_node = 4
# filter_opt = False

# for FE case study
case_name = 'FE'
dispatch_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/Dispatch_data_FE_Dispatch_whole.csv'
input_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/sweep_parameters_results_FE_whole.h5'
rev_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/FE_revenue.csv'
num_clusters = 20
num_sims = 400
input_layer_node = 4
filter_opt = True

model_save_path = f'{case_name}_case_study/{case_name}_revenue_3layers'
param_save_path = f'{case_name}_case_study/{case_name}_revenue_params_3layers.json'

# read simulation data
print('Read simulation data')
simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_name)

NNtrainer_rev = TrainNNSurrogates(simulation_data, rev_data_path, filter_opt)
# set the model type.
NNtrainer_rev.model_type = 'revenue'
x,y = NNtrainer_rev._transform_dict_to_array()

rev_surrogate = keras.models.load_model(model_save_path)
with open(param_save_path) as f:
    NN_param = json.load(f)

xm = NN_param['xm_inputs']
xstd = NN_param['xstd_inputs']
ym = NN_param['y_mean']
ystd = NN_param['y_std']

x_scaled = (x - xm)/xstd
pred_y = rev_surrogate.predict(x_scaled)
pred_y_unscaled = pred_y*ystd + ym

# calculate R2
ypredict = pred_y_unscaled.transpose()
SS_tot = np.sum(np.square(y.transpose() - ym))
SS_res = np.sum(np.square(y.transpose() - ypredict))
R2 = 1 - SS_res/SS_tot
print(R2)

# font1 = {'family' : 'Times New Roman',
#     'weight' : 'normal',
#     'size'   : 18,
#     }

# fig, axs = plt.subplots()
# fig.text(0.0, 0.5, 'Predicted revenue/$', va='center', rotation='vertical',font = font1)
# fig.text(0.4, 0.05, 'True revenue/$', va='center', rotation='horizontal',font = font1)
# fig.set_size_inches(10,10)

# yt = y.transpose()
# yp = pred_y_unscaled.transpose()

# axs.scatter(yt,yp,color = "green",alpha = 0.5)
# axs.plot([min(yt),max(yt)],[min(yt),max(yt)],color = "black")
# axs.set_title(f'Revenue',font = font1)
# axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt),0.75*max(yt)),font = font1)    

# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.tick_params(direction="in",top=True, right=True)

# plt.savefig(f'{case_name}_revenue_R2.jpg', dpi =300)

df_pred_y = pd.DataFrame(pred_y_unscaled)
result_path = F'{case_name}_predicted_revenue_3layers.csv'
df_pred_y.to_csv(result_path, index=True)