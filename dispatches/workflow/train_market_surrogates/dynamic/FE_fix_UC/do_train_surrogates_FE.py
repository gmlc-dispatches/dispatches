#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
import os
from Simulation_Data_FE import SimulationData
from Time_Series_Clustering_FE import TimeSeriesClustering
import sys 
sys.path.append("..") 
from Train_NN_Surrogates import TrainNNSurrogates
from tslearn.utils import to_time_series_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    os.chdir('..')
    current_path = os.getcwd()

    # for FE case study
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
    scaled_dispatch_dict = simulation_data._scale_data()

    df_separate_cf = pd.read_csv('FE_separate_FIX_UC/separate_cf.csv')
    diff_hours = []
    # diff_day_count = []
    for index, row in df_separate_cf.iterrows():
        old_data = scaled_dispatch_dict[index]
        new_data = row.to_numpy()[1:]
    #     # slice them into days
    #     num_days = 366
    #     diff_day = 0
    #     for day in range(num_days):
    #         old_day = old_data[day*24:(day+1)*24]
    #         new_day = new_data[day*24:(day+1)*24]
    #         if (sum(old_day) - sum(new_day)) >= 0.01:
    #             diff_day += 1
    #     diff_day_count.append(diff_day)

        
        diff = old_data - new_data
        count = 0
        for j in diff:
            if j >= 0.01:
                count += 1
        diff_hours.append(count)
    
    # print(sum(diff_hours)/400/24/366)

    # dis_cost = np.array([40.71699147,41.71699147,42.71699147,43.71699147,45.71699147,50.71699147,55.71699147,60.71699147,65.71699147,70.71699147])
    # stor_size = np.array([15,30,45,60,75,90,105,120,135,150])
    # c = 0
    # heatmap = np.zeros((len(dis_cost),len(stor_size)))
    # for p in range(len(stor_size)):
    #     for h in range(len(dis_cost)):
    #         heatmap[p,h] = diff_hours[c]
    #         c += 1
    # fig, ax = plt.subplots(figsize =(16,9))
    # im = ax.imshow(heatmap)

    # # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(dis_cost)), labels=dis_cost)
    # ax.set_yticks(np.arange(len(stor_size)), labels=stor_size)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    # # Loop over data dimensions and create text annotations.
    # for i in range(len(stor_size)):
    #     for j in range(len(dis_cost)):
    #         text = ax.text(j, i, np.round(heatmap[i, j],2),
    #                     ha="center", va="center", color="r")

    # ax.set_title("Number of different capacity factor hours")
    # fig.tight_layout()
    # # plt.show()
    # plt.savefig('dev_heapmap_hour.jpg',dpi = 300)
        


    # print('Start Time Series Clustering')
    # clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data, filter_opt)
    # # # model = clusteringtrainer.clustering_data_kmeans()
    # path = os.path.join(current_path, f'{case_type}_case_study', 'FE_result_400years_20clusters_OD.json')
    # # clusteringtrainer.save_clustering_model(model,path)
    # clusteringtrainer.find_dispatch_max_min(path)
    # mileage = clusteringtrainer._find_max_min_mileage(path)
    # arg = np.argmin(mileage[4])

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


    # TrainNNSurrogates, revenue
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
    # model_df = NNtrainer_df.train_NN_frequency([input_layer_node,75,75,75,22])
    # NN_frequency_model_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency')
    # NN_frequency_param_path = os.path.join(current_path, f'{case_type}_case_study', f'{case_type}_{num_clusters}clusters_dispatch_frequency_params.json')
    # NNtrainer_df.save_model(model_df, NN_frequency_model_path, NN_frequency_param_path)
    # NNtrainer_df.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'new_{case_type}_frequency')



if __name__ == "__main__":
    main()