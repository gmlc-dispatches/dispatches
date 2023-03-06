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
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# from dispatches.workflow.train_market_surrogates.dynamic.Wind_PEM.clustering_wind_dispatch import ClusteringDispatchWind
from dispatches.workflow.train_market_surrogates.dynamic.Wind_PEM.clustering_dispatch_pem_cf_wind import ClusteringDispatchWind
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData


def get_params(case_type):

    file_path = os.path.realpath(__file__)
    current_path, filename = os.path.split(file_path)

    if case_type == 'RE':
        # load the keras surrogate model (RE)
        
        # dispatch + wind
        # surrogate_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_wind_20', 'RE_H2_dispatch_surrogate_model_20')
        # surrogate_param_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_wind_20', 'RE_H2_dispatch_surrogate_param_20.json')
        # clustering_kmean_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_wind_20', 'RE_224years_20clusters_OD.json')

        # dispatch + exceed wind
        # surrogate_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_exceed_wind_20', 'RE_H2_dispatch_surrogate_model_dis_pem_20')
        # surrogate_param_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_exceed_wind_20', 'RE_H2_dispatch_surrogate_param_dis_pem_20.json')
        # clustering_kmean_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_exceed_wind_20', 'RE_224years_20clusters_Dispatch_PEM.json')

        # dispatch + pem_cf
        # surrogate_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_pem_cf_20', 'RE_H2_dispatch_surrogate_model_dp_cf_20')
        # surrogate_param_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_pem_cf_20', 'RE_H2_dispatch_surrogate_param_dp_cf_20.json')
        # clustering_kmean_path = os.path.join(current_path, 'Wind_PEM', 'dispatch_pem_cf_20', 'RE_224years_20clusters_Dispatch_PEM_cf.json')

        # pem_cf only
        surrogate_path = os.path.join(current_path, 'Wind_PEM', 'PEM_H2_REVENUE_surrogate', 'RE_H2_pem_cf_only_surrogate_model')
        surrogate_param_path = os.path.join(current_path, 'Wind_PEM', 'PEM_H2_REVENUE_surrogate', 'RE_H2_pem_cf_only_surrogate_param.json')
        clustering_kmean_path = os.path.join(current_path, 'Wind_PEM', 'PEM_H2_REVENUE_surrogate', 'RE_224years_20clusters_Dispatch_PEM_cf.json') # not needed in this model
        
        input_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', 'datasets', 'results_renewable_sweep_Wind_H2', 'sweep_parameters_results_RE_H2_whole.h5')

        surrogate_path_dict = {}
        surrogate_path_dict['surrogate_path'] = surrogate_path
        surrogate_path_dict['surrogate_param_path'] = surrogate_param_path
        surrogate_path_dict['clustering_kmean_path'] = clustering_kmean_path
        surrogate_path_dict['input_data_path'] = input_data_path
        surrogate_path_dict['case_type'] = 'RE'

        # Prescient data path (RE)
        num_sims = 224
        num_clusters = 20
        dispatch_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', 'datasets', 'results_renewable_sweep_Wind_H2', 'Dispatch_data_RE_H2_whole.csv')
        wind_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', 'datasets', 'results_renewable_sweep_Wind_H2', 'Real_Time_wind_hourly.csv')

        sweep_param_dict = {}
        sweep_param_dict['num_sims'] = num_sims
        sweep_param_dict['num_clusters'] = num_clusters
        sweep_param_dict['dispatch_data_path'] = dispatch_data_path
        sweep_param_dict['input_data_path'] = input_data_path
        sweep_param_dict['wind_data_path'] = wind_data_path
        sweep_param_dict['case_type'] = 'RE'

    else:
        # load the keras surrogate model (NE)
        surrogate_path = os.path.join(current_path, 'NE_case_study', 'NE_30clusters_dispatch_frequency')
        surrogate_param_path = os.path.join(current_path, 'NE_case_study', 'NE_30clusters_dispatch_frequency_params.json')
        clustering_kmean_path = os.path.join(current_path, 'NE_case_study','NE_result_192years_30clusters_OD.json')
        input_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', 'datasets', 'results_nuclear_sweep', 'sweep_parameters_results_NE_whole.h5')

        surrogate_path_dict = {}
        surrogate_path_dict['surrogate_path'] = surrogate_path
        surrogate_path_dict['surrogate_param_path'] = surrogate_param_path
        surrogate_path_dict['clustering_kmean_path'] = clustering_kmean_path
        surrogate_path_dict['input_data_path'] = input_data_path
        surrogate_path_dict['case_type'] = 'NE'

        # Prescient data path (NE)
        num_sims = 192
        num_clusters = 32
        dispatch_data_path = os.path.join(current_path, '..', '..', '..', '..', '..', 'datasets', 'results_nuclear_sweep', 'Dispatch_data_NE_whole.csv')
        wind_data_path = 'None'

        sweep_param_dict = {}
        sweep_param_dict['num_sims'] = num_sims
        sweep_param_dict['num_clusters'] = num_clusters
        sweep_param_dict['dispatch_data_path'] = dispatch_data_path
        sweep_param_dict['input_data_path'] = input_data_path
        sweep_param_dict['wind_data_path'] = wind_data_path
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
    clustering_kmean_path = surrogate_path_dict['clustering_kmean_path']
    input_data_path = surrogate_path_dict['input_data_path']
    case_type = surrogate_path_dict['case_type']

    X = read_inputs_to_array(input_data_path)

    dispatch_surrogate = keras.models.load_model(surrogate_path)
    with open(surrogate_param_path) as f:
        NN_param = json.load(f)

    # load the clustering data
    with open(clustering_kmean_path,'rb') as f:
        dispatch_kmean = json.load(f)
    centers = np.array(dispatch_kmean['model_params']['cluster_centers_'])
    centers_dict = {}
    for i in range(len(centers)):
        centers_dict[i] = centers[i]

    xm = np.array(NN_param['xm_inputs'])
    xstd = np.array(NN_param['xstd_inputs'])
    ym = np.array(NN_param['ws_mean'])
    ystd = np.array(NN_param['ws_std'])

    # loop over the input variable combinations
    surrogate_year_cf_dict = {}
    surrogate_second_term_dict = {}
    surrogate_pem_dict = {}

    if case_type == 'RE':
        for i in range(len(X)):
            # scale data
            x_scaled = np.array([(X[i] - xm)/xstd])
            # use NN predict the dispatch frequency
            pred_y = dispatch_surrogate.predict(x_scaled,verbose = 0)
            # unscale the predict results
            pred_y_unscaled = pred_y*ystd + ym
            # output is 2D. 
            ws = pred_y_unscaled[0]

            # sum(representative_day_cf[j]*ws[j] for j in num_clusters)
            total_dispatch_cf_surrogate = 0
            total_second_term_surrogate = 0
            total_pem_power = 0
            for j in range(len(ws)):
            # #     This is for dispatch + wind
        #         rep_day_dispatch = centers_dict[j][0]
        #         rep_day_wind = centers_dict[j][1]
        #         total_dispatch = ws[j]*366*sum(rep_day_dispatch)/24/366
        #         total_wind = ws[j]*366*sum(rep_day_wind)/24/366
        #         exceed_power = rep_day_wind - rep_day_dispatch
        #         pem_power = np.clip(exceed_power, 0, X[i][1]/847)    # pem max power may be lower than the exceed wind power
        #         total_pem =  ws[j]*366*sum(pem_power)/24/366
        #         total_dispatch_cf_surrogate += total_dispatch
        #         total_second_term_surrogate += total_wind
        #         total_pem_power += total_pem
        
        #     surrogate_year_cf_dict[i] = total_dispatch_cf_surrogate
        #     surrogate_second_term_dict[i] = total_second_term_surrogate     # second term here is the averge wind cf, no need to return, but we record it here.
        #     surrogate_pem_dict[i] = total_pem_power     # this term needs to return, the pem cf.

        # return surrogate_year_cf_dict, surrogate_pem_dict

            # #   This is for dispatch + exceed_elec
        #         rep_day_dispatch = centers_dict[j][0]
        #         exceed_elec = centers_dict[j][1]
        #         real_pem_elec = np.clip(exceed_elec, 0, X[i][1]/847)
        #         total_dispatch = ws[j]*366*sum(rep_day_dispatch)/24/366
        #         total_pem_elec = ws[j]*366*sum(real_pem_elec)/24/366
        #         total_dispatch_cf_surrogate += total_dispatch
        #         total_second_term_surrogate += total_pem_elec

        #     surrogate_year_cf_dict[i] = total_dispatch_cf_surrogate
        #     surrogate_second_term_dict[i] = total_second_term_surrogate     # second term here is pem cf.

        # return surrogate_year_cf_dict, surrogate_second_term_dict

            # #   This is for dispatch + pem_elec_cf 
                rep_day_dispatch = centers_dict[j][0]
                pem_day_cf = centers_dict[j][1]
                total_dispatch = ws[j]*366*sum(rep_day_dispatch)/24/366
                total_pem_elec = ws[j]*366*sum(pem_day_cf)/24/366*X[i][1]/847       # scale to 847 based data. Consistent with the previous two cases
                total_dispatch_cf_surrogate += total_dispatch
                total_second_term_surrogate += total_pem_elec

            surrogate_year_cf_dict[i] = total_dispatch_cf_surrogate
            surrogate_second_term_dict[i] = total_second_term_surrogate     # second term here is pem cf.

        return surrogate_year_cf_dict, surrogate_second_term_dict

    else:
        for i in range(len(X)):
            # scale data
            x_scaled = np.array([(X[i] - xm)/xstd])
            # use NN predict the dispatch frequency
            pred_y = dispatch_surrogate.predict(x_scaled,verbose = 0)
            # unscale the predict results
            pred_y_unscaled = pred_y*ystd + ym
            # output is 2D. 
            ws = pred_y_unscaled[0]
            # NE, we have filters and 0/1 scaling.
            # In the calculation of NE case, we define the net power to grid instead of capacity factor. 
            NE_pmax = 400    # MW
            NE_pmin = 400*(1-X[i][1])
            for j in range(len(ws)-2):
                rep_day_dispatch = centers_dict[j]*(NE_pmax-NE_pmin)+NE_pmin    # centers_dict[j] is 2d np.array: np.array([[cf1],[cf2],...[cf24]])
                # reminder: for the nn with filter in clustering, ws[0] and ws[1] are frequency for zero and full days. 
                total_dispatch = ws[j+1]*366*sum(rep_day_dispatch)
                total_dispatch_cf_surrogate += total_dispatch

            # add zero day and full day.
            total_dispatch_cf_surrogate += ws[0]*366*NE_pmin*24
            total_dispatch_cf_surrogate += ws[-1]*366*NE_pmax*24

            surrogate_year_cf_dict[i] = total_dispatch_cf_surrogate[0]    # total_dispatch_cf_surrogate is np.array with 1 elemets, so take the value here
            surrogate_second_term_dict = 'None'

        return surrogate_year_cf_dict, surrogate_second_term_dict


def calculate_sweep_year_capacity_factor(sweep_param_dict):
    '''
    return the dict, keys are index of sweep, values are year capacity factors.
    '''
    if sweep_param_dict['case_type'] == 'RE':
        num_sims = sweep_param_dict['num_sims']
        num_clusters = sweep_param_dict['num_clusters']
        dispatch_data_path = sweep_param_dict['dispatch_data_path']
        wind_data_path = sweep_param_dict['wind_data_path']
        input_data_path = sweep_param_dict['input_data_path']
        # read the input data in order to get the pem Pmax for each simulation
        X = read_inputs_to_array(input_data_path)

        # use the original version, since we only use it to give us the dispatch profile, which is the same across all versions
        dw = ClusteringDispatchWind(dispatch_data_path, wind_data_path, '303_WIND_1', num_sims, num_clusters)   
        dispatch_array = dw.read_data()

        # read whole year wind data and scale to [0,1]
        total_wind_profile = pd.read_csv(wind_data_path)
        selected_wind_data = total_wind_profile['303_WIND_1'].to_numpy()
        scaled_selected_wind_data = selected_wind_data/847

        sweep_year_cf_dict = {}
        sweep_wind_cf_dict = {}
        sweep_pem_cf_dict = {}

        for i in range(len(dispatch_array)):
            year_dispatch_cf = dispatch_array[i]/847
            sweep_year_cf_dict[i] = sum(year_dispatch_cf)/24/366
            sweep_wind_cf_dict[i] = sum(scaled_selected_wind_data)/24/366       # still, record but do not return
            sweep_pem_power = np.clip(selected_wind_data - dispatch_array[i], 0, X[i][1])      # X[i][1] is the pem Pmax for each simulation, here the result is not scaled
            sweep_pem_cf_dict[i] = sum(sweep_pem_power)/24/366/X[i][1]      # average hour cf in each simulation

        return sweep_year_cf_dict, sweep_pem_cf_dict

    else: 
        num_sims = sweep_param_dict['num_sims']
        num_clusters = sweep_param_dict['num_clusters']
        dispatch_data_path = sweep_param_dict['dispatch_data_path']
        input_data_path = sweep_param_dict['input_data_path']
        case_type = sweep_param_dict['case_type']

        sd = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
        
        NE_dispatch_dict = sd._dispatch_dict
        sweep_year_cf_dict = {}
        for i in NE_dispatch_dict:
            year_dispatch = NE_dispatch_dict[i]
            sweep_year_cf_dict[i] = sum(year_dispatch)

        return sweep_year_cf_dict


def make_dispatch_power_heatmap(case_type, sweep_year_cf_dict, surrogate_year_cf_dict):
    '''
    Make the heatmap for annual dispatch capacity factor 
    '''
    if case_type == 'NE':
        pem_ratio = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5])
        pem_bid = np.array([15,20,25,30,35,40])
        result_dict = {}
        rf_max_lmp_pair = [(10,500),(10,1000),(15,500),(15,1000)]
        c = 0
        for p in rf_max_lmp_pair:
            ratio_arrray = np.zeros((len(pem_ratio),len(pem_bid)))
            for i in range(len(pem_ratio)):
                for j in range(len(pem_bid)):
                    r = surrogate_year_cf_dict[c]/sweep_year_cf_dict[c]
                    ratio_arrray[i][j] = r
                    c += 1
            result_dict[p] = ratio_arrray
        
        for p in result_dict:
            fig, ax = plt.subplots(figsize =(16,9))
            im = ax.imshow(result_dict[p].T)

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
            fig.tight_layout()
            plt.savefig(f'{case_type} dispatch_cf_ratio {p[0],p[1]}', dpi =300)
    
    else:
        pem_bid = np.array([15,20,25,30,35,40,45])
        pem_power = np.array([127.5,169.4,211.75,254.1,296.45,338.8,381.15,423.5])
        result_dict = {}
        rf_max_lmp_pair = [(10,500),(10,1000),(15,500),(15,1000)]
        c = 0
        for p in rf_max_lmp_pair:
            ratio_arrray = np.zeros((len(pem_power),len(pem_bid)))
            surrogate_dispatch_array = np.zeros((len(pem_power),len(pem_bid)))
            sweep_dispatch_array = np.zeros((len(pem_power),len(pem_bid)))
            for i in range(len(pem_power)):
                for j in range(len(pem_bid)):
                    r = surrogate_year_cf_dict[c]/sweep_year_cf_dict[c]
                    ratio_arrray[i][j] = r
                    surrogate_dispatch_array[i][j] = surrogate_year_cf_dict[c]*24*366*847*1e-3      # GWh
                    sweep_dispatch_array[i][j] = sweep_year_cf_dict[c]*24*366*847*1e-3      # GWh
                    c += 1
            result_dict[p] = [ratio_arrray, surrogate_dispatch_array, sweep_dispatch_array]
        
        for p in result_dict:
            fig, axs = plt.subplots(1,3, figsize =(16,9))
            title = ['surrogate dispatch cf/sweep dispatch cf', 'surrogate dispatch power/GWh', 'sweep dispatch power/GWh']
            for m in range(len(axs)):
                im = axs[m].imshow(result_dict[p][m].T, origin='lower')

                # Show all ticks and label them with the respective list entries
                axs[m].set_xticks(np.arange(len(pem_power)), labels=pem_power)
                axs[m].set_yticks(np.arange(len(pem_bid)), labels=pem_bid)
                axs[m].set_xlabel('pem power/MW')
                axs[m].set_ylabel('pem bid/$')
                # Rotate the tick labels and set their alignment.
                plt.setp(axs[m].get_xticklabels(), rotation=45, ha="right",
                            rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                for i in range(len(pem_power)):
                    for j in range(len(pem_bid)):
                        if m == 0:
                            text = axs[m].text(i, j, np.round(result_dict[p][m][i, j],3),
                                            ha="center", va="center", color="r")
                        else:
                            text = axs[m].text(i, j, np.round(result_dict[p][m][i, j],1),
                                            ha="center", va="center", color="r")

                axs[m].set_title(f"{case_type} " + title[m] + f" ({p[0]}, {p[1]})")
                                
            fig.tight_layout()
            plt.savefig(f'{case_type} dispatch_cf_ratio {p[0],p[1]}_dis_pem', dpi =300)
    
    return


def calculate_pem_only_surrogate(surrogate_path_dict):

    surrogate_path = surrogate_path_dict['surrogate_path']
    surrogate_param_path = surrogate_path_dict['surrogate_param_path']
    input_data_path = surrogate_path_dict['input_data_path']

    # read input data to array
    input_data_array = read_inputs_to_array(input_data_path)

    pem_cf_surrogate = keras.models.load_model(surrogate_path)
    with open(surrogate_param_path) as f:
        NN_param = json.load(f)

    xm = np.array(NN_param['xm_inputs'])
    xstd = np.array(NN_param['xstd_inputs'])
    ym = np.array(NN_param['y_mean'])
    ystd = np.array(NN_param['y_std'])

    pem_surrogate_cf_dict = {}

    for i in range(len(input_data_array)):
        # scale data
        x_scaled = np.array([(input_data_array[i] - xm)/xstd])
        # use NN predict the dispatch frequency
        pred_y = pem_cf_surrogate.predict(x_scaled,verbose = 0)
        # unscale the predict results
        pred_y_unscaled = pred_y*ystd + ym
        # results is 2D like np.array([[0.5]])
        y = pred_y_unscaled[0]
        
        pem_surrogate_cf_dict[i] = y[0]

    return pem_surrogate_cf_dict, input_data_array


def make_h2_revenue_heat_map(pem_surrogate_cf_dict, pem_sweep_cf_dict, input_data_array):
    '''
    Three plots for hydrogen revenue.
    first one is surrogate_pem_cf/sweep_pem_cf for each simulation.
    second one is surrogate pem revenue.
    third one is sweep pem revenue.
    '''

    pem_bid = np.array([15,20,25,30,35,40,45])
    pem_power = np.array([127.5,169.4,211.75,254.1,296.45,338.8,381.15,423.5])
    result_dict = {}
    rf_max_lmp_pair = [(10,500),(10,1000),(15,500),(15,1000)]
    c = 0
    h2_price = 3
    h2_conversion = 54.953
    for p in rf_max_lmp_pair:
        ratio_arrray = np.zeros((len(pem_power),len(pem_bid)))
        surrogate_rev_array = np.zeros((len(pem_power),len(pem_bid)))
        sweep_rev_array = np.zeros((len(pem_power),len(pem_bid)))
        for i in range(len(pem_power)):
            for j in range(len(pem_bid)):
                surrogate_rev_array[i][j] = pem_surrogate_cf_dict[c]*24*366*input_data_array[c][1]/h2_conversion*h2_price*1e-3
                sweep_rev_array[i][j] = pem_sweep_cf_dict[c]*24*366*input_data_array[c][1]/h2_conversion*h2_price*1e-3
                ratio_arrray[i][j] = surrogate_rev_array[i][j]/sweep_rev_array[i][j]
                c += 1
        result_dict[p] = [ratio_arrray, surrogate_rev_array, sweep_rev_array]
    
    for p in result_dict:
        fig, (ax0,ax1,ax2) = plt.subplots(1, 3, figsize =(16,9))
        im0 = ax0.imshow(result_dict[p][0].T, origin='lower')

        # Show all ticks and label them with the respective list entries
        ax0.set_xticks(np.arange(len(pem_power)), labels=pem_power)
        ax0.set_yticks(np.arange(len(pem_bid)), labels=pem_bid)
        ax0.set_xlabel('pem power/MW')
        ax0.set_ylabel('pem bid/$')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax0.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(pem_power)):
            for j in range(len(pem_bid)):
                text0 = ax0.text(i, j, np.round(result_dict[p][0][i, j],3),
                                ha="center", va="center", color="r")

        ax0.set_title(f"{case_type} surrogate pem cf/sweep pem cf, ({p[0]}, {p[1]})")

        im1 = ax1.imshow(result_dict[p][1].T,origin='lower')
        # Show all ticks and label them with the respective list entries
        ax1.set_xticks(np.arange(len(pem_power)), labels=pem_power)
        ax1.set_yticks(np.arange(len(pem_bid)), labels=pem_bid)
        ax1.set_xlabel('pem power/MW')
        ax1.set_ylabel('pem bid/$')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(pem_power)):
            for j in range(len(pem_bid)):
                text1 = ax1.text(i, j, np.round(result_dict[p][1][i, j],1),
                                ha="center", va="center", color="r")

        ax1.set_title(f"{case_type} surrogate H2 revenue, M$, ({p[0]}, {p[1]})")

        im2 = ax2.imshow(result_dict[p][2].T,origin='lower')
        # Show all ticks and label them with the respective list entries
        ax2.set_xticks(np.arange(len(pem_power)), labels=pem_power)
        ax2.set_yticks(np.arange(len(pem_bid)), labels=pem_bid)
        ax2.set_xlabel('pem power/MW')
        ax2.set_ylabel('pem bid/$')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
                    rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(pem_power)):
            for j in range(len(pem_bid)):
                text2 = ax2.text(i, j, np.round(result_dict[p][2][i, j],1),
                                ha="center", va="center", color="r")

        ax2.set_title(f"{case_type} sweep H2 revenue M$, ({p[0]}, {p[1]})")

        fig.tight_layout()
        plt.savefig(f'{case_type} pem_revenue {p[0],p[1]}', dpi =300)

    return

# case_type = 'NE'
# num_sims = 192
case_type = 'RE'
num_sims = 224

surrogate_path_dict, sweep_param_dict = get_params(case_type)
# sweep_year_cf_dict, sweep_pem_cf_dict = calculate_sweep_year_capacity_factor(sweep_param_dict)
# surrogate_year_cf_dict, surrogate_second_term_dict = calculate_surrogate_year_capacity_factor(surrogate_path_dict)

# Read input data to array
X  = read_inputs_to_array(sweep_param_dict['input_data_path'])

# test_h2_Rev(surrogate_second_term_dict, sweep_pem_cf_dict, X)   
# for 2D clustering models
# make_dispatch_power_heatmap(case_type, sweep_year_cf_dict, surrogate_year_cf_dict)
# make_h2_revenue_heat_map(surrogate_second_term_dict, sweep_pem_cf_dict, X)

# for pem_only surrogate
task = ClusteringDispatchWind(sweep_param_dict['dispatch_data_path'], sweep_param_dict['input_data_path'], sweep_param_dict['wind_data_path'], '303_WIND_1', num_sims, 20)
dispatch_array = task.read_data()
pem_sweep_cf = task.calculate_PEM_cf(dispatch_array)
pem_sweep_cf_dict = {}
for i in range(len(pem_sweep_cf)):
    pem_sweep_cf_dict[i] = sum(pem_sweep_cf[i])/24/366

pem_surrogate_cf_dict, input_data_array = calculate_pem_only_surrogate(surrogate_path_dict)

make_h2_revenue_heat_map(pem_surrogate_cf_dict, pem_sweep_cf_dict, input_data_array)

