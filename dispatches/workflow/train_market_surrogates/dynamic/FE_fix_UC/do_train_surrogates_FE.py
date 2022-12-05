import os
from Simulation_Data_FE import SimulationData
from Time_Series_Clustering_FE import TimeSeriesClustering
import sys 
sys.path.append("..") 
from Train_NN_Surrogates import TrainNNSurrogates
from tslearn.utils import to_time_series_dataset


def main():
    os.chdir('..')
    current_path = os.getcwd()

    # for NE case study
    dispatch_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/Dispatch_data_FE_Dispatch_whole.csv'
    input_data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/sweep_parameters_results_FE_whole.h5'
    case_type = 'FE'
    num_clusters = 20
    num_sims = 400
    input_layer_node = 4
    filter_opt = True

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    # scaled_dispatch_dict = simulation_data._scale_data()
    # print(scaled_dispatch_dict[90][1340:1370])

    print('Start Time Series Clustering')
    clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data, filter_opt)
    # model = clusteringtrainer.clustering_data_kmeans()
    path = os.path.join(current_path, f'{case_type}_case_study', 'FE_result_400years_20clusters_OD.json')
    # clusteringtrainer.save_clustering_model(model,path)
    clusteringtrainer.find_dispatch_max_min(path)


    # day_dataset = clusteringtrainer._transform_data()
    # clusteringtrainer._divide_data(day_dataset)
    # # clusteringtrainer._transform_data()
    # clustering_model_use, clustering_model_not_use= clusteringtrainer.clustering_data_kmeans()
    # clustering_model = clusteringtrainer.clustering_data_kmedoids()
    # clusteringtrainer.plot_results_kmedoid(clustering_model,i)
    # clustering_result_path_use = os.path.join(current_path, f'{case_type}_case_study', f'New_try_use_{case_type}_result_{num_sims}years_{num_clusters}clusters_OD.json')
    # clustering_result_path_not_use = os.path.join(current_path, f'{case_type}_case_study', f'New_try_not_use_{case_type}_result_{num_sims}years_{num_clusters}clusters_OD.json')
    # result_path_use = clusteringtrainer.save_clustering_model(clustering_model_use, fpath = clustering_result_path_use)
    # result_path_not_use = clusteringtrainer.save_clustering_model(clustering_model_not_use, fpath = clustering_result_path_not_use)
    
    # plot clustering figures
    # clusteringtrainer.plot_results_2D(path)

    # check results
    # clusteringtrainer.check_results(clustering_result_path_use, clustering_result_path_not_use)

    # # plot boxplots
    # clusteringtrainer.box_plots(result_path)
    
    # # plot cluster centers in one figure.
    # clusteringtrainer.plot_centers(result_path)


    # # TrainNNSurrogates, revenue
    # print('Start train revenue surrogate')
    # data_path = '../../../../../datasets/results_fossil_sweep_revised_fixed_commitment/FE_revenue.csv'
    # NNtrainer_rev = TrainNNSurrogates(simulation_data, data_path, filter_opt)
    # model_rev = NNtrainer_rev.train_NN_revenue([input_layer_node,100,100,1])
    # # save to given path
    # NN_rev_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_revenue')
    # NN_rev_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_revenue_params.json')
    # NNtrainer_rev.save_model(model_rev, NN_rev_model_path, NN_rev_param_path)
    # NNtrainer_rev.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue_plot.jpg')

    # TrainNNSurrogates, dispatch frequency
    # print('Start train dispatch frequency surrogate')
    # model_type = 'frequency'
    # clustering_model_path = os.path.join(current_path,'FE_case_study','FE_result_400years_20clusters_OD.json')
    # NNtrainer_df = TrainNNSurrogates(simulation_data, clustering_model_path, filter_opt = filter_opt)
    # NNtrainer_df.case_type = 'frequency'
    # NNtrainer_df._read_clustering_model(clustering_model_path)
    # df_dict = NNtrainer_df._generate_label_data()
    # print(df_dict)
    # model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,22])
    # NN_frequency_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency')
    # NN_frequency_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json')
    # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    # NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')



if __name__ == "__main__":
    main()