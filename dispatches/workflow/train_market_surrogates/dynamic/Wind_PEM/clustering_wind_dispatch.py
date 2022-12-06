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
This code do clustering over wind+dispatch data for the new parameter sweep wind+H2

Use this script instead of automation workflow is because of the difficulty in adding multi-dimension clustering methods
in the automation workflow. 

filter out dispatch profile days with capacity factor always = 0/1 days

clustering data (dispatch, wind, pv), 3*24 array

dispatch: (d1, d2,..., d24)

wind: (w1, w2,..., w24)

'''

class ClusteringDispatchWind:
    def __init__(self, dispatch_data, wind_data, wind_gen, years, num_clusters, metric = 'euclidean'):
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

        selected_wind_data = selected_wind_data/self.wind_gen_pmax

        wind_data = []
        time_len = 24
        day_num = int(len(selected_wind_data)/time_len)
        for i in range(day_num):
            wind_data.append(np.array(selected_wind_data[i*24:(i+1)*24]))

        # wind_data will have shape of (366, 24) with all data scaled by p_wind_max
        return wind_data


    def transform_data(self, dispatch_array, wind_data, filters = True):

        '''
        shape the data to the format that tslearn can read.

        Arguments:
            dispatch data in the shape of numpy array. (Can be obtained from self.read_data())

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

        full_day = []
        zero_day = []

        for year,idx in zip(dispatch_years, dispatch_years_index):
            # scale by the p_max
            pmax_of_year = self.wind_gen_pmax
            scaled_year = year/pmax_of_year

            # slice the year data into day data(24 hours a day)

            if filters == True:
            # filter out full/zero capacity days
                for i in range(day_num):
                    dispatch_day_data = scaled_year[i*time_len:(i+1)*time_len]
                    # count the day of full/zero capacity factor.
                    # Sepearte the data out. np.shape(zero/full_day) = (num_days, 2, 24)
                    if sum(dispatch_day_data) == 0:
                        zero_day.append([dispatch_day_data,wind_data[i]])
                    elif sum(dispatch_day_data) == 24:
                        full_day.append([dispatch_day_data,wind_data[i]])
                    else:
                        # np.shape(datasets) = (num_days, 2, 24)
                        # (wind(1*24), dispatch(1*24))
                        datasets.append([dispatch_day_data,wind_data[i]])
            # no filter
            else:
                for i in range(day_num):
                    dispatch_day_data = scaled_year[i*time_len:(i+1)*time_len]
                    datasets.append([dispatch_day_data,wind_data[i]])
        zero_full_days = [zero_day, full_day]
        # use tslearn package to form the correct data structure.
        train_data = to_time_series_dataset(datasets)

        return train_data, zero_full_days
        

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

        km = TimeSeriesKMeans(n_clusters = clusters, metric = self.metric, random_state = 42)
        labels = km.fit_predict(train_data)

        if save_index == True:
            path0 = os.getcwd()
            result_path = os.path.join(path0, fname)
            km.to_json(result_path)

        return labels


    # def cluster_data_scikit(self, train_data, clusters):
    #     sk_train_data = to_sklearn_dataset(train_data)
    #     print(sk_train_data)
    #     km = KMeans(n_clusters=clusters, random_state=0)
    #     km.fit_predict(sk_train_data)
    #     return km.cluster_centers_



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


    def plot_results(self, result_path, train_data, num_clusters, idx):
        
        '''
        Plot the result data. this is for 2-d clustering data.

        Arguments: 

            result_path: the path of json file that has clustering results

            idx: int, the index that of the cluster center

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

        f,(ax1,ax2) = plt.subplots(2,1)
        for data in label_data_dict[idx]:
            ax1.plot(time_length, data[0], '--', c='g', alpha=0.05)
            ax2.plot(time_length, data[1], '--', c='g', alpha=0.05)

        ax1.plot(time_length, centers_dict[idx][0], '-', c='r', alpha=1.0)
        ax2.plot(time_length, centers_dict[idx][1], '-', c='r', alpha=1.0)
        ax1.set_ylabel('Capacity factor',font = font1)
        ax2.set_ylabel('Capacity factor',font = font1)
        ax1.set_xlabel('Time(h)',font = font1)
        ax2.set_xlabel('Time(h)',font = font1)
        figname = f'clustering_figures/RE_result_{num_clusters}clusters_cluster{idx}.jpg'
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
        cluster_max_dispatch = {}
        cluster_min_dispatch = {}
        cluster_median_dispatch = {}
        cluster_max_wind = {}
        cluster_min_wind = {}
        cluster_median_wind = {}

        for idx in range(self.num_clusters):
            cluster_max_dispatch[idx] = []
            cluster_min_dispatch[idx] = []
            cluster_median_dispatch[idx] = []
            cluster_max_wind[idx] = []
            cluster_min_wind[idx] = []
            cluster_median_wind[idx] = []
            sum_dispatch_data = []
            for data in label_data_dict[idx]:
                sum_dispatch_data.append(np.sum(data[0]))

            median_index = np.argsort(sum_dispatch_data)[len(sum_dispatch_data) // 2]
            cluster_max_dispatch[idx].append(label_data_dict[idx][np.argmax(sum_dispatch_data)][0].tolist())
            cluster_max_wind[idx].append(label_data_dict[idx][np.argmax(sum_dispatch_data)][1].tolist())
            cluster_min_dispatch[idx].append(label_data_dict[idx][np.argmin(sum_dispatch_data)][0].tolist())
            cluster_min_wind[idx].append(label_data_dict[idx][np.argmin(sum_dispatch_data)][1].tolist())
            cluster_median_dispatch[idx].append(label_data_dict[idx][median_index][0].tolist())
            cluster_median_wind[idx].append(label_data_dict[idx][median_index][1].tolist())

        # with open('dispatch_max_min_median.json', 'w') as f:
        #     json.dump({'max_dispatch':cluster_max_dispatch, 'min_dispatch': cluster_min_dispatch, 'median_dispatch':cluster_median_dispatch,\
        #         'max_wind':cluster_min_wind, 'min_wind':cluster_min_wind, 'median_wind':cluster_median_wind}, f)

        # for idx in range(self.num_clusters):
        #     f,(ax0,ax1) = plt.subplots(2,1)
        #     for data in label_data_dict[idx]:
        #         ax0.plot(time_length, data[0], '--', c='g', alpha=0.05)
        #         ax1.plot(time_length, data[1], '--', c='g', alpha=0.05)
        #     ax0.plot(time_length, centers_dict[idx][0], '-', c='r', alpha=1.0, label = 'mean')
        #     ax1.plot(time_length, centers_dict[idx][1], '-', c='r', alpha=1.0, label = 'mean')
        #     ax0.plot(time_length, cluster_max_dispatch[idx][0], '-', c='b', alpha=1.0, label = 'max')
        #     ax1.plot(time_length, cluster_max_wind[idx][0], '-', c='b', alpha=1.0, label = 'max')
        #     ax0.plot(time_length, cluster_min_dispatch[idx][0], '-', c='k', alpha=1.0, label = 'min')
        #     ax1.plot(time_length, cluster_min_wind[idx][0], '-', c='k', alpha=1.0, label = 'min')
        #     ax0.plot(time_length, cluster_median_dispatch[idx][0], '-', c='m', alpha=1.0, label = 'median')
        #     ax1.plot(time_length, cluster_median_wind[idx][0], '-', c='m', alpha=1.0, label = 'median')
        #     ax0.set_ylabel('Capacity factor',font = font1)
        #     ax0.set_xlabel('Time(h)',font = font1)
        #     ax1.set_ylabel('Capacity factor',font = font1)
        #     ax1.set_xlabel('Time(h)',font = font1)
        #     ax0.legend()
        #     ax1.legend()
        #     ax0.set_title('Dispatch Profile')
        #     ax1.set_title('Wind Profile')

        #     figname = f'clustering_figures/RE_dispatch_min_max_{idx}.jpg'
        #     plt.savefig(figname, dpi = 300)

        return [cluster_max_dispatch, cluster_min_dispatch, cluster_median_dispatch], [cluster_max_wind, cluster_min_wind, cluster_median_wind]


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
            figname = f'clustering_figures/RE_wind_dispatch_check_{idx}.jpg'
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
    wind_gen = '303_WIND_1'
    tsa_task = ClusteringDispatchWind(dispatch_data, wind_data, wind_gen, years, num_clusters)
    dispatch_array = tsa_task.read_data()
    wind_data_list = tsa_task.read_wind_data()
    train_data,day_01 = tsa_task.transform_data(dispatch_array, wind_data_list, filters = filters)

    fname = f'../RE_case_study/RE_224years_20clusters_OD.json'
    # labels = tsa_task.cluster_data(train_data, num_clusters, fname, save_index = True)

    if filters == True:
        print('full capacity days = {}'.format(len(day_01[1])))
        print('zero capacity days = {}'.format(len(day_01[0])))
    else:
        print('No filters')

    # tsa_task.find_wind_max_min(fname, train_data)
    # cluster_max_wind, cluster_max_dispatch = tsa_task.find_dispatch_max_min(fname,train_data)
    # tsa_task.wind_dispatch_check(fname)
    tsa_task.cluster_analysis(fname, train_data)
    tsa_task.wind_dispatch_check(fname, )
    # # count errors
    # err_count = 0
    # bad_index = []
    # for i in range(len(train_data)):
    #     dispatch = train_data[i][0]
    #     wind = train_data[i][1]
    #     for t in range(24):
    #         diff = wind[t] - dispatch[t] 
    #         if diff < -0.01:
    #             err_count += 1
    #             bad_index.append([i//366,i%366])
    # print(err_count)





if __name__ == '__main__':
    main()
