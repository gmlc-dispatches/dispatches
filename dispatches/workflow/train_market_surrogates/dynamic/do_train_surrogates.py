import os
from Simulation_Data import SimulationData
from Train_NN_Surrogates import TrainNNSurrogates
from Time_Series_Clustering import TimeSeriesClustering

def main():

    current_path = os.getcwd()

    # for RE_H2 case study
    # dispatch_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2_new/Dispatch_data_RE_H2_whole.xlsx'
    # input_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2_new/sweep_parameters_results_RE_H2_whole.h5'
    # case_type = 'RE'
    # num_clusters = 20
    # num_sims = 224
    # input_layer_node = 4

    # for NE case study
    # dispatch_data_path = '../../../../../datasets/results_nuclear_sweep/Dispatch_data_NE_whole.xlsx'
    # input_data_path = '../../../../../datasets/results_nuclear_sweep/sweep_parameters_results_nuclear_whole.h5'
    # case_type = 'NE'
    # num_clusters = 30
    # num_sims = 192
    # input_layer_node = 4
    # filter_opt = True

    # for FE case study (use the test dataset)
    dispatch_data_path = '../../../../../datasets/results_fossil_sweep/Dispatch_data_FE_whole.xlsx'
    input_data_path = '../../../../../datasets/results_fossil_sweep/sweep_parameters_results_FE_whole.h5'
    case_type = 'FE'
    num_clusters = 30
    num_sims = 1065*4
    input_layer_node = 5
    filter_opt = True

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    # # for RE_H2 case study clustering need to be done in 2-d (dispatch + wind), so I do this in another script.
    # print('Start Time Series Clustering')
    # clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data)
    # clustering_model = clusteringtrainer.clustering_data()
    # clustering_result_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_result_{num_sims}years_{num_clusters}clusters.json')
    # result_path = clusteringtrainer.save_clustering_model(clustering_model, fpath = clustering_result_path)
    # for i in range(num_clusters):
    #     clusteringtrainer.plot_results(result_path, i)
    # outlier_count = clusteringtrainer.box_plots(result_path)
    # clusteringtrainer.plot_centers(result_path)


    # TrainNNSurrogates, revenue
    print('Start train revenue surrogate')
    model_type = 'revenue'
    clustering_model_path = 'placeholder'
    NNtrainer_rev = TrainNNSurrogates(simulation_data, clustering_model_path, model_type, filter_opt)
    model_rev = NNtrainer_rev.train_NN([input_layer_node,100,100,1])
    NN_rev_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_revenue')
    NN_rev_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_revenue_params.json')
    NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
    NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue_plot.jpg')

    # TrainNNSurrogates, dispatch frequency
    print('Start train dispatch frequency surrogate')
    model_type = 'frequency'
    clustering_model_path = clustering_result_path
    NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, model_type, filter_opt = True)
    model_df = NNtrainer_df.train_NN([input_layer_node,75,75,75,32])
    NN_frequency_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency')
    NN_frequency_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json')
    NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'{case_type}_frequency')



if __name__ == "__main__":
    main()