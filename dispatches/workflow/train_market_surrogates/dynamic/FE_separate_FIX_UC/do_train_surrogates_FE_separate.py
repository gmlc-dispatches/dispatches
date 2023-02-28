import os
from Simulation_Data_FE_separate import SimulationData
from Time_Series_Clustering_FE_separate import TimeSeriesClustering
from Train_NN_Surrogates_separate import TrainNNSurrogates
from tslearn.utils import to_time_series_dataset
import pathlib
import numpy as np


def main():
    os.chdir('..')
    # for FE case study
    dispatch_data_path_generator = pathlib.Path('../../../../../datasets/results_fossil_sweep_revised_fixed_commitment_disaggregated/Dispatch_generator_data_FE_separate_whole.csv')
    dispatch_data_path_storage = pathlib.Path('../../../../../datasets/results_fossil_sweep_revised_fixed_commitment_disaggregated/Dispatch_storage_data_FE_separate_whole.csv')
    input_data_path = pathlib.Path('../../../../../datasets/results_fossil_sweep_revised_fixed_commitment_disaggregated/sweep_parameters_results_FE_whole.h5')
    case_type = 'FE'
    num_clusters = 20
    num_sims = 400
    input_layer_node = 4
    filter_opt = True

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data_generator = SimulationData(dispatch_data_path_generator, input_data_path, num_sims, case_type)
    simulation_data_storage = SimulationData(dispatch_data_path_storage, input_data_path, num_sims, case_type)
    os.chdir('FE_separate_FIX_UC')
    tsa_separate = TimeSeriesClustering(num_clusters, simulation_data_generator, simulation_data_storage, filter_opt)
    # clustering_model = tsa_separate.clustering_data_kmeans()
    # tsa_separate._transform_data()
    clustering_result_path = f'{case_type}_result_{num_sims}years_{num_clusters}clusters_OD_separate.json'
    # tsa_separate.save_clustering_model(clustering_model, fpath = clustering_result_path)
    # cluster_95_dispatch_index, cluster_5_dispatch_index, cluster_median_dispatch_index = tsa_separate.find_dispatch_max_min(clustering_result_path)
    # tsa_separate.find_target_gen_storage_data(cluster_95_dispatch_index, cluster_5_dispatch_index, cluster_median_dispatch_index)
    
    
    # train dispatch frequency surrogate model
    print('Start train dispatch frequency surrogate')

    NNtrainer_df = TrainNNSurrogates(simulation_data_generator, simulation_data_storage, clustering_result_path, filter_opt = filter_opt)
    model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,75,75,22])
    # NN_frequency_model_path = f'{case_type}_{num_clusters}clusters_dispatch_frequency_separate'
    # NN_frequency_param_path = f'{case_type}_{num_clusters}clusters_dispatch_frequency_separate_params.json'
    # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    # NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')

    # save the data to csv
    # year_cf_list = []
    # df_index = []
    # for i in index:
    #     year_cf = scaled_dispatch_dict[i]
    #     df_year_cf = pd.DataFrame(year_cf).T
    #     year_cf_list.append(df_year_cf)
    #     df_index.append(i)

    # data_output = pd.concat(year_cf_list)
    # data_output.index = df_index
    # data_output.to_csv('FE_separate_FIX_UC/separate_cf.csv', index = True)


if __name__ == "__main__":
    main()