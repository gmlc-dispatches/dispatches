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
from sklearn.cluster import KMeans
from tslearn.utils import to_sklearn_dataset
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans,silhouette_score
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,12)
import os
import re
import json

'''
This code do clustering over dispatch + pem electricity data for the new parameter sweep wind+H2

Use this script instead of automation workflow is because of the difficulty in adding multi-dimension clustering methods
in the automation workflow. 

dispatch: (d1, d2, ..., d24)

wind: (w1, w2,..., w24)

PEM electricity = (w1-d1, w2-d2, ..., w24-d24)

real_pem_elec = np.clip(PEM electricity, 0, pem_pmax)

clustering over (dispatch, real_pem_elec)
'''


class ClusteringDispatchWind:
    def __init__(self, dispatch_data, input_data, wind_data, wind_gen, years, num_clusters, metric = 'euclidean'):
        '''
        Initializes the bidder object.

        Arguments:
            dispatch_data: csv files with the dispatch power data

            wind_data: csv files with wind profile

            metric: distance metric (“euclidean” or “dtw”).

            years: The size for the clustering dataset.

            wind_gen: name of wind generator

        Return:
            None
        '''
        self.dispatch_data = dispatch_data
        self.input_data = input_data
        self.wind_data = wind_data
        self.metric = metric
        self.years = int(years)
        self.num_clusters = num_clusters
        self.wind_gen = wind_gen

        # make a dict key = generator name, value = pmax
        _wind_gen_pmax = {}
        _wind_gen_name = ['309_WIND_1', '317_WIND_1', '303_WIND_1', '122_WIND_1']
        _win_gen_pmax_list = [148.3, 799.1, 847, 713.5]


        for name, pmax in zip(_wind_gen_name, _win_gen_pmax_list):
            _wind_gen_pmax[name] = pmax

        # check the wind generator and pv generator name are correct.
        # Assign pmax accroding to generators.
        if self.wind_gen in _wind_gen_name:
            self.wind_gen_pmax = _wind_gen_pmax[self.wind_gen]
        else:
            raise NameError("wind generator name {} is invaild.".format(self.wind_gen))   

    def read_data(self):

        '''
        read clustering data from dispatch csv files
        
        Aruguments:
            None

        Return: 
            numpy array with dispatch data.
        '''

        # One sim year data is one row, read the target rows.
        df_dispatch = pd.read_csv(self.dispatch_data, nrows = self.years)

        # drop the first column
        df_dispatch_data = df_dispatch.iloc[: , 1:]

        # the first column is the run_index. Put them in an array
        df_index = df_dispatch.iloc[:,0]
        run_index = df_index.to_numpy(dtype = str)

        # In csv files, every run is not in sequence from 0 to 64999. 
        # run indexs are strings of 'run_xxxx.csv', make xxxx into a list of int
        self.index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            self.index.append(int(index_num))

        # transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
        dispatch_array = df_dispatch_data.to_numpy(dtype = float)

        return dispatch_array


    def read_wind_data(self):

        '''
        The length of the dispatch profile and wind profile are in the same length
        which are 366*24 = 8784 days.
        '''
        wind_file = self.wind_data

        total_wind_profile = pd.read_csv(wind_file)
        selected_wind_data = total_wind_profile[self.wind_gen].to_numpy()

        scaled_selected_wind_data = selected_wind_data/self.wind_gen_pmax

        wind_data = []
        time_len = 24
        day_num = int(len(scaled_selected_wind_data)/time_len)
        for i in range(day_num):
            wind_data.append(np.array(scaled_selected_wind_data[i*24:(i+1)*24]))

        # wind_data will have shape of (366, 24) with all data scaled by p_wind_max
        return wind_data


    def calculate_PEM_cf(self, dispatch_array):
        '''
        calculate the pem cf for each simulation.
        '''
        # read the input data
        df_input_data = pd.read_hdf(self.input_data)
        # return the number of columns in the df, that is the dimension of the input space. 
        num_col = df_input_data.shape[1]
        num_row = df_input_data.shape[0]

        # drop the first column, which is the indexes
        input_array = df_input_data.iloc[list(range(num_row)), list(range(1,num_col))].to_numpy()
        
        # read the wind profile
        wind_file = self.wind_data

        total_wind_profile = pd.read_csv(wind_file)
        # store the wind data in array
        wind_data = total_wind_profile[self.wind_gen].to_numpy()

        real_pem_elec_cf = []
        # calcluate the pem power for each simulation
        for idx in range(len(dispatch_array)):
            pem_elec = wind_data - dispatch_array[idx]
            # sometimes we have small pem and pem_elec > pem_pmax
            # input_array[idx][1] is the pem max power for simulation idx.
            real_pem_elec = np.clip(pem_elec, 0, input_array[idx][1])
            real_pem_elec_cf.append(real_pem_elec/input_array[idx][1])
        
        return real_pem_elec_cf


    def transform_data(self, dispatch_array, real_pem_elec_cf, filters = True):

        '''
        shape the data to the format that tslearn can read.

        (dispatch_day_data, pem_electricity), pem_electricity = wind_day_data - dispatch_day_data

        Arguments:

        Return:
            train_data: np.arrya for the tslearn package. Dimension = (self.years*364, 24, 1)
            data of full/zero days: [zero_day,full_day]
        '''
        


        datasets = []
        # number of hours in a representative day
        time_len = 24

        # should have 366 day in a year in our simulation
        day_num = int(np.round(len(dispatch_array[0])/time_len))
        self.day_num = day_num

        # How many years are there in dispatch_array is depending on the self.dispatch_data.
        # We use the target number of years (self.years) to do the clustering
        dispatch_years = dispatch_array[0:self.years]

        # Need to have the index to do scaling by pmax. 
        dispatch_years_index = self.index[0:self.years]

        for dispatch_data, pem_cf in zip(dispatch_years, real_pem_elec_cf):
            # scale by the p_max
            pmax_of_year = self.wind_gen_pmax
            scaled_year = dispatch_data/pmax_of_year
 
            for i in range(day_num):
                dispatch_day_data = scaled_year[i*time_len:(i+1)*time_len]
                real_pem_day_elec = pem_cf[i*time_len:(i+1)*time_len]
                datasets.append([dispatch_day_data, real_pem_day_elec*1e-1])  # add weights to the data.

        # use tslearn package to form the correct data structure.
        train_data = to_time_series_dataset(datasets)

        return train_data
        

    def cluster_data(self, train_data, clusters, fname, save_index = False):

        '''
        cluster the data. Save the model to a json file. 
        
        Arguments:
            train_data: from self.transform_data
            clusters: number of clusters specified
            fname: to save file names
            save_index: bool, when True, save the cluster center results in json file.
        
        return:
            label of the data
        '''

        km = TimeSeriesKMeans(n_clusters = clusters, metric = self.metric, random_state = 42, max_iter = 100, verbose=1)
        labels = km.fit_predict(train_data)

        if save_index == True:
            path0 = os.getcwd()
            result_path = os.path.join(path0, fname)
            km.to_json(result_path)

        return labels


    def get_cluster_centers(self, result_path):

        '''
        Get the cluster centers.

        Arguments:

            result_path: the path of clustering model

        Returns:

            centers_list: {cluster_center:[results]} 
        '''

        with open(result_path, 'r') as f:
            cluster_results = json.load(f)
        
        centers = np.array(cluster_results['model_params']['cluster_centers_'])

        centers_dict = {}
        for i in range(len(centers)):
            centers_dict[i] = centers[i]

        return centers_dict


    def _summarize_results(self, result_path, train_data):

        '''
        Summarize the results from the clustering

        Arguments:

            result_path: the path of json file that has clustering results
        
        Returns:

            label_data_dict: dictionary that has the label data {cluster_number:[train_data]}

        '''

        with open(result_path, 'r') as f:
            cluster_results = json.load(f)
        
        labels = cluster_results['model_params']['labels_']

        label_data_dict = {}
        for idx,lb in enumerate(labels):
            if lb not in label_data_dict:
                label_data_dict[lb] = []
                label_data_dict[lb].append(train_data[idx])
            else:
                label_data_dict[lb].append(train_data[idx])

        return label_data_dict


    def plot_results(self, result_path, train_data, num_clusters):
        
        '''
        Plot the result data. this is for 2-d clustering data.

        Arguments: 

            result_path: the path of json file that has clustering results

        Returns:

            None
        '''

        label_data_dict = self._summarize_results(result_path, train_data)
        centers_dict = self.get_cluster_centers(result_path)

        time_length = range(24)
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

        for idx in range(self.num_clusters):
            f,(ax1,ax2) = plt.subplots(2,1)
            for data in label_data_dict[idx]:
                ax1.plot(time_length, data[0], '--', c='g', alpha=0.05)
                ax2.plot(time_length, data[1]*1e1, '--', c='g', alpha=0.05)

            ax1.plot(time_length, centers_dict[idx][0], '-', c='r', alpha=1.0)
            ax2.plot(time_length, centers_dict[idx][1]*1e1, '-', c='r', alpha=1.0)
            ax1.set_ylabel('Capacity factor',font = font1)
            ax2.set_ylabel('Capacity factor',font = font1)
            ax1.set_xlabel('Time(h)',font = font1)
            ax2.set_xlabel('Time(h)',font = font1)
            ax1.set_title('dispatch power capacity factor', font = font1)
            ax2.set_title('PEM power capacity factor', font = font1)
            figname = f'clustering_figures/RE_result_{num_clusters}clusters_dispatch_pem_cf_cluster_weighted{idx}.jpg'
            plt.savefig(figname, dpi = 300)

        return


    def find_dispatch_max_min(self, result_path, train_data):
        '''
        Find the max and min wind profile within the cluster.  
        '''
        label_data_dict = self._summarize_results(result_path, train_data)
        centers_dict = self.get_cluster_centers(result_path)
        
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        time_length = range(24)
        cluster_median_dispatch = {}
        cluster_95_dispatch = {}
        cluster_5_dispatch = {}
        cluster_95_wind = {}
        cluster_5_wind = {}
        cluster_median_wind = {}

        for idx in range(self.num_clusters):
            sum_dispatch_data = []
            for data in label_data_dict[idx]:
                sum_dispatch_data.append(np.sum(data[0]))

            median_index = np.argsort(sum_dispatch_data)[len(sum_dispatch_data) // 2]
            quantile_95_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.95)]
            quantile_5_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.05)]
            cluster_95_dispatch[idx] = label_data_dict[idx][quantile_95_index][0].tolist()
            cluster_5_dispatch[idx] = label_data_dict[idx][quantile_5_index][0].tolist()
            cluster_95_wind[idx] = label_data_dict[idx][quantile_95_index][1].tolist()
            cluster_5_wind[idx] = label_data_dict[idx][quantile_5_index][1].tolist()
            cluster_median_dispatch[idx] = label_data_dict[idx][median_index][0].tolist()
            cluster_median_wind[idx] = label_data_dict[idx][median_index][1].tolist()

        with open('dispatch_95_5_median_new.json', 'w') as f:
            json.dump({'cluster_95_dispatch':cluster_95_dispatch, 'cluster_5_dispatch':cluster_5_dispatch, 'median_dispatch':cluster_median_dispatch,\
                'cluster_95_wind':cluster_95_wind, 'cluster_5_wind':cluster_5_wind, 'median_wind':cluster_median_wind}, f)

        for idx in range(self.num_clusters):
            f,(ax0,ax1) = plt.subplots(2,1)
            for data in label_data_dict[idx]:
                ax0.plot(time_length, data[0], '--', c='g', alpha=0.05)
                ax1.plot(time_length, data[1], '--', c='g', alpha=0.05)
            cf_center_0 = np.sum(centers_dict[idx][0])/24
            cf_center_1 = np.sum(centers_dict[idx][1])/24
            ax0.plot(time_length, centers_dict[idx][0], '-', c='r', linewidth=3, alpha=1.0, label = f'representative ({round(cf_center_0,3)})')
            ax1.plot(time_length, centers_dict[idx][1], '-', c='r', linewidth=3, alpha=1.0, label = f'representative ({round(cf_center_1,3)})')
            cf_95_0 = np.sum(cluster_95_dispatch[idx])/24
            cf_95_1 = np.sum(cluster_95_wind[idx])/24
            ax0.plot(time_length, cluster_95_dispatch[idx], '-', c='brown', linewidth=3, alpha=1.0, label = f'95 quantile ({round(cf_95_0,3)})')
            ax1.plot(time_length, cluster_95_wind[idx], '-', c='brown', linewidth=3, alpha=1.0, label = f'95 quantile ({round(cf_95_1,3)})')
            cf_5_0 = np.sum(cluster_5_dispatch[idx])/24
            cf_5_1 = np.sum(cluster_5_wind[idx])/24
            ax0.plot(time_length, cluster_5_dispatch[idx], '-', c='pink', linewidth=3, alpha=1.0, label = f'5 quantile ({round(cf_5_0,3)})')
            ax1.plot(time_length, cluster_5_wind[idx], '-', c='pink', linewidth=3, alpha=1.0, label = f'5 quantile ({round(cf_5_1,3)})')
            cf_med_0 = np.sum(cluster_median_dispatch[idx])/24
            cf_med_1 = np.sum(cluster_median_wind[idx])/24
            ax0.plot(time_length, cluster_median_dispatch[idx], '-', c='k', linewidth=3, alpha=1.0, label = f'median ({round(cf_med_0,3)})')
            ax1.plot(time_length, cluster_median_wind[idx], '-', c='k', linewidth=3, alpha=1.0, label = f'median ({round(cf_med_1,3)})')
            ax0.set_ylabel('Capacity factor',font = font1)
            ax0.set_xlabel('Time(h)',font = font1)
            ax1.set_ylabel('Capacity factor',font = font1)
            ax1.set_xlabel('Time(h)',font = font1)
            ax0.legend()
            ax1.legend()
            ax0.set_title('Dispatch Profile')
            ax1.set_title('Wind Profile')

            figname = f'clustering_figures/RE_dispatch_95_5_median_{self.num_clusters}_{idx}.jpg'
            plt.savefig(figname, dpi = 300)

        # return [cluster_max_dispatch, cluster_min_dispatch, cluster_median_dispatch], [cluster_max_wind, cluster_min_wind, cluster_median_wind]


    def wind_dispatch_check(self, result_path, dispatch_result, wind_result):
        centers_dict = self.get_cluster_centers(result_path)
        time_length = range(24)
        d_max,d_min,d_med = dispatch_result
        w_max,w_min,w_med = wind_result

        for idx in range(self.num_clusters):
            f,(ax0,ax1,ax2,ax3) = plt.subplots(2,2)
            ax0.plot(time_length, centers_dict[idx][0], label = 'Dispatch')
            ax0.plot(time_length, centers_dict[idx][1], label = 'Wind')
            ax1.plot(time_length, d_max[idx][0], label = 'Dispatch')
            ax1.plot(time_length, w_max[idx][0], label = 'Wind')
            ax2.plot(time_length, d_min[idx][0], label = 'Dispatch')
            ax2.plot(time_length, w_min[idx][0], label = 'Wind')
            ax3.plot(time_length, d_med[idx][0], label = 'Dispatch')
            ax3.plot(time_length, w_med[idx][0], label = 'Wind')
            plt.xlabel('Time/h')
            plt.ylabel('Capacity factor')
            plt.legend()
            figname = f'clustering_figures/new_RE_wind_dispatch_check_{idx}.jpg'
            plt.savefig(figname, dpi = 300)

        return


    def cluster_analysis(self, result_path, train_data):

        label_data_dict = self._summarize_results(result_path, train_data)
        centers_dict = self.get_cluster_centers(result_path)

        for idx in range(self.num_clusters):
            sum_dispatch_capacity_factor = []
            sum_wind_capacity_factor = []
            for data in label_data_dict[idx]:
                sum_dispatch_capacity_factor.append(np.sum(data[0])/24*10)
                sum_wind_capacity_factor.append(np.sum(data[1])/24*10)
            fig,ax = plt.subplots()
            bins = list(range(11))
            ax.hist(sum_dispatch_capacity_factor, bins = bins, density = True, label = 'Dispatch')
            ax.hist(sum_wind_capacity_factor, bins = bins, density = True, label = 'Wind')
            ax.set_ylabel('Probability Density')
            ax.set_xlabel('Day Capacity Factors')
            ax.set_title(f'Dispatch/Wind histogram cluster_{idx}')
            ax.legend()
            plt.savefig(f'histogram_cluster_{idx}.jpg')


def main():

    metric = 'euclidean'
    years = 224

    num_clusters = 20
    filters = False

    dispatch_data = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_Dispatch_whole.csv'
    wind_data = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Real_Time_wind_hourly.csv'
    input_data = '../../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    wind_gen = '303_WIND_1'
    tsa_task = ClusteringDispatchWind(dispatch_data, input_data, wind_data, wind_gen, years, num_clusters)
    dispatch_array = tsa_task.read_data()
    real_pem_elec_cf = tsa_task.calculate_PEM_cf(dispatch_array)

    train_data = tsa_task.transform_data(dispatch_array, real_pem_elec_cf, filters = filters)

    fname = f'RE_224years_{num_clusters}clusters_Dispatch_PEM_cf_weighted.json'
    labels = tsa_task.cluster_data(train_data, num_clusters, fname, save_index = True)
    tsa_task.plot_results(fname, train_data, num_clusters)
    
    # if filters == True:
    #     print('full capacity days = {}'.format(len(day_01[1])))
    #     print('zero capacity days = {}'.format(len(day_01[0])))
    # else:
    #     print('No filters')

    # print(np.shape(train_data))
    # tsa_task.find_dispatch_max_min(fname,train_data)
    # tsa_task.wind_dispatch_check(fname)
    # tsa_task.cluster_analysis(fname, train_data)


if __name__ == '__main__':
    main()
