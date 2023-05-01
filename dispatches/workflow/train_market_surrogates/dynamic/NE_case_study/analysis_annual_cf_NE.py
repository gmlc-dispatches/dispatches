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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.NE_case_study.Train_NN_Surrogates_steady_state import TrainNNSurrogates


def get_params(case_type = 'RE'):

    current_path = os.getcwd()

    # load the keras surrogate model (NE)
    
    # static clustering
    surrogate_path = os.path.join(current_path, 'steady_state', 'tanh_25_25', 'NE_steady_state')
    surrogate_param_path = os.path.join(current_path, 'steady_state', 'tanh_25_25', 'NE_steady_state_params.json')
    input_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', '..', 'datasets','results_nuclear_sweep','sweep_parameters_results_NE_whole.h5')

    surrogate_path_dict = {}
    surrogate_path_dict['surrogate_path'] = surrogate_path
    surrogate_path_dict['surrogate_param_path'] = surrogate_param_path
    surrogate_path_dict['input_data_path'] = input_data_path
    surrogate_path_dict['case_type'] = 'NE'

    # Prescient data path (RE)
    num_sims = 192
    dispatch_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', '..', 'datasets','results_nuclear_sweep','Dispatch_data_NE_whole.csv')

    sweep_param_dict = {}
    sweep_param_dict['num_sims'] = num_sims
    sweep_param_dict['dispatch_data_path'] = dispatch_data_path
    sweep_param_dict['input_data_path'] = input_data_path
    sweep_param_dict['case_type'] = 'NE'

    return surrogate_path_dict, sweep_param_dict


def read_inputs_to_array(input_data_path):
    
    df_input_data = pd.read_hdf(input_data_path)
    num_col = df_input_data.shape[1]
    num_row = df_input_data.shape[0]
    X = df_input_data.iloc[list(range(num_row)),list(range(1,num_col))].to_numpy()

    return X


def calculate_surrogate_year_capacity_factor(surrogate_path_dict):
    '''
    return the dict, keys are index of sweep, values are year capacity factors.
    '''
    surrogate_path = surrogate_path_dict['surrogate_path']
    surrogate_param_path = surrogate_path_dict['surrogate_param_path']
    input_data_path = surrogate_path_dict['input_data_path']
    case_type = surrogate_path_dict['case_type']

    X = read_inputs_to_array(input_data_path)

    cf_surrogate = keras.models.load_model(surrogate_path)
    with open(surrogate_param_path) as f:
        NN_param = json.load(f)

    xm = np.array(NN_param['xm_inputs'])
    xstd = np.array(NN_param['xstd_inputs'])
    ym = np.array(NN_param['y_mean'])
    ystd = np.array(NN_param['y_std'])

    # loop over the input variable combinations
    surrogate_dispatch_cf_dict = {}


    for i in range(len(X)):
        # scale data
        x_scaled = np.array([(X[i] - xm)/xstd])
        # use NN predict the dispatch frequency
        pred_y = cf_surrogate.predict(x_scaled,verbose = 0)
        # unscale the predict results
        pred_y_unscaled = pred_y*ystd + ym
        # output is 2D. 
        y = pred_y_unscaled[0]

        for j in y:
            surrogate_dispatch_cf_dict[i] = j

    return surrogate_dispatch_cf_dict


def calculate_sweep_year_capacity_factor(sweep_param_dict):
    '''
    return the dict, keys are index of sweep, values are year capacity factors.
    '''

    num_sims = sweep_param_dict['num_sims']
    dispatch_data_path = sweep_param_dict['dispatch_data_path']
    input_data_path = sweep_param_dict['input_data_path']
    # read the input data in order to get the pem Pmax for each simulation
    X = read_inputs_to_array(input_data_path)

    sim_data = SimulationData(dispatch_data_path, input_data_path, num_sims, 'NE')
    
    dispatch_data_dict, input_data_dict = sim_data.read_data_to_dict()

    sweep_dispatch_cf_dict = {}
    # i is the index of sweep simulations
    for i in dispatch_data_dict:
        rt_dispatch = dispatch_data_dict[i]
        rt_dispatch_cf = np.sum(rt_dispatch)/(400*len(rt_dispatch))
        sweep_dispatch_cf_dict[i] = rt_dispatch_cf

    return sweep_dispatch_cf_dict


def make_dispatch_power_heatmap(case_type, sweep_dispatch_cf_dict, surrogate_year_cf_dict):
    '''
    Make the heatmap for annual dispatch capacity factor 
    '''

    pem_ratio = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5])
    pem_bid = np.array([15,20,25,30,35,40])/20
    result_dict = {}
    rf_max_lmp_pair = [(10,500),(10,1000),(15,500),(15,1000)]
    c = 0
    for p in rf_max_lmp_pair:
        ratio_arrray = np.zeros((len(pem_ratio),len(pem_bid)))
        for i in range(len(pem_ratio)):
            for j in range(len(pem_bid)):
                r = surrogate_year_cf_dict[c]/sweep_dispatch_cf_dict[c]
                ratio_arrray[i][j] = r
                c += 1
        result_dict[p] = ratio_arrray
    
    for p in result_dict:
        fig, ax = plt.subplots(figsize =(16,9))
        im = ax.imshow(result_dict[p].T,origin='lower')

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(pem_ratio)), labels=pem_ratio)
        ax.set_yticks(np.arange(len(pem_bid)), labels=pem_bid)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(pem_ratio)):
            for j in range(len(pem_bid)):
                text = ax.text(i, j, np.round(result_dict[p][i, j],5),
                                ha="center", va="center", color="r")

        ax.set_title(f"{case_type} surrogate_year_cf/sweep_year_cf, rf = {p[0]}, max_lmp = {p[1]}")
        ax.set_xlabel('PEM/NPP ratio')
        ax.set_ylabel('H2 Price ($/kg)')
        fig.tight_layout()
        plt.savefig(f'{case_type} dispatch_cf_ratio {p[0],p[1]}', dpi =300)
    
    
    return


case_type = 'NE'
num_sims = 192


surrogate_path_dict, sweep_param_dict = get_params(case_type)
sweep_year_cf_dict = calculate_sweep_year_capacity_factor(sweep_param_dict)
surrogate_year_cf_dict = calculate_surrogate_year_capacity_factor(surrogate_path_dict)

# Read input data to array
X  = read_inputs_to_array(sweep_param_dict['input_data_path'])

make_dispatch_power_heatmap(case_type, sweep_year_cf_dict, surrogate_year_cf_dict)


