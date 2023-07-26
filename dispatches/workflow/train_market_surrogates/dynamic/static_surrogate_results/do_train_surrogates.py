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

from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.Simulation_Data_subscenario import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.Train_NN_Surrogates_subscenario import TrainNNSurrogates
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.Time_Series_Clustering_subscenario import TimeSeriesClustering
import pathlib
from dispatches_data.api import path

# this is for training revenue/dynamic dispatch frequency surrogates

def main():
    # for NE case study
    path_to_data_package = path("dynamic_sweep")
    case_type = "NE"
    model_type = "revenue"

    if case_type == "NE":
        dispatch_data_path = path_to_data_package / "NE" / "Dispatch_data_NE_Dispatch_whole.csv"
        input_data_path = path_to_data_package / "NE" / "sweep_parameters_results_NE_whole.h5"
        case_type = 'NE'
        num_clusters = 30
        num_sims = 192
        input_layer_node = 4
        filter_opt = True

    # for FE case study
    if case_type == "FE":
        dispatch_data_path = path_to_data_package / "FE" / "Dispatch_generator_data_FE_separate_whole.csv"
        input_data_path = path_to_data_package / "FE" / "sweep_parameters_results_FE_whole.h5"
        case_type = 'FE'
        num_clusters = 20
        num_sims = 400
        input_layer_node = 4
        filter_opt = True

    # for RE case study
    if case_type == "RE":
        dispatch_data_path = path_to_data_package / "RE" / "Dispatch_data_RE_H2_Dispatch_whole.csv"
        input_data_path = path_to_data_package / "RE" / "sweep_parameters_results_RE_H2_whole.h5"
        case_type = 'RE'
        num_clusters = 20
        num_sims = 224
        input_layer_node = 4
        filter_opt = False

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)

    # print('Start Time Series Clustering')
    # clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data, filter_opt)
    # clustering_model = clusteringtrainer.clustering_data_kmeans()
    clustering_result_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_sims}years_{num_clusters}clusters_OD.json'))
    # clusteringtrainer.save_clustering_model(clustering_model, fpath = clustering_result_path)
    # # plot results
    # for i in range(num_clusters):
    #     clusteringtrainer.plot_results(clustering_result_path, i)
    # clusteringtrainer.box_plots(clustering_result_path)
    

    # TrainNNSurrogates, revenue
    if model_type == "revenue":
        print('Start train revenue surrogate')
        hidden_nodes = 25
        hidden_layers = 2
        if case_type == "RE":
            data_path = path_to_data_package / "RE" / "RE_H2_RT_revenue.csv"
            NN_rev_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', 'RT_revenue', f'{case_type}_RT_revenue_{hidden_layers}_{hidden_nodes}'))
            NN_rev_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', 'RT_revenue', f'{case_type}_RT_revenue_params_{hidden_layers}_{hidden_nodes}.json'))
        if case_type == "NE":
            data_path = path_to_data_package / "NE" / "NE_revenue.csv"
            NN_rev_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', 'revenue', f'{case_type}_revenue_{hidden_layers}_{hidden_nodes}'))
            NN_rev_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', 'revenue', f'{case_type}_revenue_params_{hidden_layers}_{hidden_nodes}.json'))
        
        NNtrainer_rev = TrainNNSurrogates(simulation_data, data_path, filter_opt)
        # model_rev = NNtrainer_rev.train_NN_revenue([input_layer_node,hidden_nodes,hidden_nodes,1])
        
        # save to given path
        NNtrainer_rev.model_type = 'revenue'
        # NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
        NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue_plot_{hidden_layers}_{hidden_nodes}.jpg')

    # TrainNNSurrogates, dispatch frequency
    if model_type == "frequency":
        print('Start train dispatch frequency surrogate')
        clustering_model_path = clustering_result_path
        NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, filter_opt = filter_opt)
        NNtrainer_df.model_type = 'frequency'
        model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,32])
        NN_frequency_model_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency'))
        NN_frequency_param_path = str(pathlib.Path.cwd().joinpath(f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json'))
        # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
        NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')


if __name__ == "__main__":
    main()