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
from dispatches.workflow.train_market_surrogates.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.Train_NN_Surrogates import TrainNNSurrogates
from dispatches.workflow.train_market_surrogates.Time_Series_Clustering import TimeSeriesClustering

def main():

    current_path = os.getcwd()

    # for NE case study
    dispatch_data_path = '../../../../../datasets/results_nuclear_sweep/Dispatch_data_NE_whole.csv'
    input_data_path = '../../../../../datasets/results_nuclear_sweep/sweep_parameters_results_NE_whole.h5'
    case_type = 'NE'
    num_clusters = 30
    num_sims = 192
    input_layer_node = 4
    filter_opt = True

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    print('Start Time Series Clustering')
    clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data, filter_opt)
    clustering_model = clusteringtrainer.clustering_data()
    clustering_result_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_result_{num_sims}years_{num_clusters}clusters_OD.json')
    result_path = clusteringtrainer.save_clustering_model(clustering_model, fpath = clustering_result_path)
    
    # plot clustering figures
    for i in range(num_clusters):
        clusteringtrainer.plot_results(result_path, i)

    # plot boxplots
    clusteringtrainer.box_plots(result_path)
    
    # plot cluster centers in one figure.
    clusteringtrainer.plot_centers(result_path)


    # TrainNNSurrogates, revenue
    print('Start train revenue surrogate')
    data_path = '../../../../../datasets/results_nuclear_sweep/NE_revenue.csv'
    NNtrainer_rev = TrainNNSurrogates(simulation_data, data_path, filter_opt)
    model_rev = NNtrainer_rev.train_NN_revenue([input_layer_node,100,100,1])
    # save to given path
    NN_rev_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_revenue')
    NN_rev_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_revenue_params.json')
    NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
    NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue_plot.jpg')

    # TrainNNSurrogates, dispatch frequency
    print('Start train dispatch frequency surrogate')
    model_type = 'frequency'
    clustering_model_path = clustering_result_path
    NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, filter_opt = True)
    model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,32])
    NN_frequency_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency')
    NN_frequency_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json')
    NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')



if __name__ == "__main__":
    main()