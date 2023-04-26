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

import os
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.Train_NN_Surrogates import TrainNNSurrogates
from dispatches.workflow.train_market_surrogates.dynamic.Time_Series_Clustering import TimeSeriesClustering
import pathlib

def main():
    # for NE case study
    dispatch_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_nuclear_sweep','Dispatch_data_NE_whole.csv'))
    input_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_nuclear_sweep','sweep_parameters_results_NE_whole.h5'))
    case_type = 'NE'
    num_clusters = 30
    num_sims = 192
    input_layer_node = 4
    filter_opt = True

    # for FE case study
    # dispatch_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_fossil_sweep_revised_fixed_commitment','Dispatch_data_FE_Dispatch_whole.csv'))
    # input_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_fossil_sweep_revised_fixed_commitment','sweep_parameters_results_FE_whole.h5'))
    # case_type = 'FE'
    # num_clusters = 20
    # num_sims = 400
    # input_layer_node = 4
    # filter_opt = True

    # for RE case study
    # dispatch_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_renewable_sweep_Wind_H2','Dispatch_data_RE_H2_Dispatch_whole.csv'))
    # input_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_renewable_sweep_Wind_H2','sweep_parameters_results_RE_H2_whole.h5'))
    # case_type = 'RE'
    # num_clusters = 20
    # num_sims = 224
    # input_layer_node = 4
    # filter_opt = False

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)


    # print('Start Time Series Clustering')
    # clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data, filter_opt)
    # clustering_model = clusteringtrainer.clustering_data_kmeans()
    clustering_result_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_sims}years_{num_clusters}clusters_OD.json'))
    # clusteringtrainer.save_clustering_model(clustering_model, fpath = clustering_result_path)
    # # plot results
    # for i in range(30):
    #     clusteringtrainer.plot_results(clustering_result_path, i)
    # clusteringtrainer.box_plots(clustering_result_path)
    

    # TrainNNSurrogates, revenue
    # print('Start train revenue surrogate')
    # data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_renewable_sweep_Wind_H2','RE_H2_revenue.csv'))
    # data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_nuclear_sweep','NE_revenue.csv'))
    # NNtrainer_rev = TrainNNSurrogates(simulation_data, data_path, filter_opt)
    # model_rev = NNtrainer_rev.train_NN_revenue([input_layer_node,75,75,75,1])
    # # save to given path
    # NN_rev_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_revenue_3layers'))
    # NN_rev_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_revenue_params_3layers.json'))
    # NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
    # NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue_plot_3layers.jpg')

    # TrainNNSurrogates, dispatch frequency
    print('Start train dispatch frequency surrogate')
    model_type = 'frequency'
    clustering_model_path = clustering_result_path
    NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, filter_opt = filter_opt)
    model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,32])
    NN_frequency_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency'))
    NN_frequency_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json'))
    # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')



if __name__ == "__main__":
    main()