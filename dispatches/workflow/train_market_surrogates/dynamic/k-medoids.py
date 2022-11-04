from sklearn_extra.cluster import KMedoids
import numpy as np
from Simulation_Data import SimulationData
from Train_NN_Surrogates import TrainNNSurrogates
from Time_Series_Clustering import TimeSeriesClustering
from tslearn.utils import to_sklearn_dataset
import matplotlib.pyplot as plt

dispatch_data_path = '../../../../../datasets/results_nuclear_sweep/Dispatch_data_NE_whole.csv'
input_data_path = '../../../../../datasets/results_nuclear_sweep/sweep_parameters_results_nuclear_whole.h5'
case_type = 'NE'
num_clusters = 30
num_sims = 192
input_layer_node = 4
filter_opt = True

print('Read simulation data')
simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data)
train_data = clusteringtrainer._transform_data()
sk_train_data = to_sklearn_dataset(train_data)
num_clusters = 30
kmedoids = KMedoids(n_clusters=num_clusters, random_state=0)
kmedoids.fit(sk_train_data)

# save centers to dict
centers_dict = {}
for i, cen in enumerate(kmedoids.cluster_centers_):
    centers_dict[i] = cen

# save labels of each data to dict
label_data_dict = {}
for i, lb in enumerate(kmedoids.labels_):
    if lb not in label_data_dict:
        label_data_dict[lb] = []
        label_data_dict[lb].append(sk_train_data[i])

    else:
        label_data_dict[lb].append(sk_train_data[i])



def plot_results(kmedoids, centers_dict, label_data_dict, idx):

    time_length = range(24)
    font1 = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 18,
    }

    f,ax1 = plt.subplots(figsize = ((16,6)))
    for data in label_data_dict[idx]:
        ax1.plot(time_length, data, '--', c='g', alpha=0.3)

    ax1.plot(time_length, centers_dict[idx], '-', c='r', alpha=1.0)
    ax1.set_ylabel('Capacity factor',font = font1)
    ax1.set_xlabel('Time(h)',font = font1)
    figname = f'NE_case_study/clustering_figures/NE_kmedoids_result_{num_clusters}clusters_cluster{idx}.jpg'
    plt.savefig(figname, dpi = 300)

    return

for i in range(20):
    plot_results(kmedoids, centers_dict, label_data_dict, i)