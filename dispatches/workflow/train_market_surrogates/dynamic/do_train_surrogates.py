import os
from Simulation_Data import SimulationData
from Train_NN_Surrogates import TrainNNSurrogates
from Time_Series_Clustering import TimeSeriesClustering

def main():

    current_path = os.getcwd()

    # for NE case study
    dispatch_data_path = '../../../../../datasets/results_nuclear_sweep/Dispatch_data_NE_whole.csv'
    input_data_path = '../../../../../datasets/results_nuclear_sweep/sweep_parameters_results_NE_whole.h5'
    rev_data_path = '../../../../../datasets/results_nuclear_sweep/NE_revenue.csv'
    
    # # for RE case study
    # dispatch_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_whole.csv'
    # input_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    # rev_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2/RE_H2_revenue.csv'

    # for FE case study
    # dispatch_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/Dispatch_data_FE_Dispatch_whole.csv'
    # input_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/sweep_parameters_results_FE_whole.h5'
    # rev_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/FE_revenue.csv'

    case_type = 'NE'
    num_clusters = 30
    num_sims = 192
    input_layer_node = 4
    filter_opt = True
    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    # print('Start Time Series Clustering')
    # clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data, filter_opt)
    # clustering_model = clusteringtrainer.clustering_data()
    # clustering_result_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_result_{num_sims}years_{num_clusters}clusters_OD.json')
    # result_path = clusteringtrainer.save_clustering_model(clustering_model, fpath = clustering_result_path)
    
    # # plot clustering figures
    # for i in range(num_clusters):
    #     clusteringtrainer.plot_results(result_path, i)

    # # plot boxplots
    # clusteringtrainer.box_plots(result_path)
    
    # # plot cluster centers in one figure.
    # clusteringtrainer.plot_centers(result_path)


    # TrainNNSurrogates, revenue
    print('Start train revenue surrogate')

    NNtrainer_rev = TrainNNSurrogates(simulation_data, rev_data_path, filter_opt)
    model_rev = NNtrainer_rev.train_NN_revenue([input_layer_node,75,75,75,1])
    # save to given path
    NN_rev_model_path = os.path.join(current_path, f'{case_type}_case_study_new', f'{case_type}_revenue_new')
    NN_rev_param_path = os.path.join(current_path, f'{case_type}_case_study_new', f'{case_type}_revenue_params_new.json')
    NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
    NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue_plot.jpg')

    # TrainNNSurrogates, dispatch frequency
    # print('Start train dispatch frequency surrogate')
    # model_type = 'frequency'
    # clustering_model_path = clustering_result_path
    # NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, filter_opt = True)
    # model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,32])
    # NN_frequency_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency')
    # NN_frequency_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json')
    # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    # NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')



if __name__ == "__main__":
    main()