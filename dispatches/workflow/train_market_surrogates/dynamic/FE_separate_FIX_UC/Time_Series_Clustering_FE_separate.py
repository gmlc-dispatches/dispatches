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

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from idaes.core.util import to_json, from_json
from sklearn_extra.cluster import KMedoids
from tslearn.utils import to_sklearn_dataset
import time
import numpy as np
import json
import re
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,9)


class TimeSeriesClustering:

    def __init__(self, num_clusters, simulation_data_gen, simulation_data_storage, filter_opt = True, metric = 'euclidean'):
        
        ''' 
        Time series clustering for the dispatch data. 

        Now only can do clustering over dispatch data.

        Arguments:
            
            simulation data: object, composition from ReadData class
            
            metric: metric for clustering, must be one of euclidean or dtw

            num_clusters: number of clusters that we want to have

            filter_opt: bool, if we are going to filter out 0/1 capacity days
        
        '''
        self.simulation_data_gen = simulation_data_gen
        self.simulation_data_storage = simulation_data_storage
        self.metric = metric
        self.num_clusters = num_clusters
        self.filter_opt = filter_opt
        # set a class property which is the time length of a day.
        self._time_length = 24


    # @property
    # def simulation_data(self):

    #     '''
    #     Porperty getter of simulation_data
        
    #     Arguments:

    #         None

    #     Returns:
    #         simulation_data
    #     '''
        
    #     return self._simulation_data


    # @simulation_data.setter
    # def simulation_data(self, value):

    #     '''
    #     Porperty setter of simulation_data
        
    #     Arguments:

    #         value: object, composition from ReadData class

    #     Returns:
    #         None
    #     '''
        
    #     if not isinstance(value, object):
    #         raise TypeError(
    #             f"The simulation_data must be an object, but {type(value)} is given."
    #         )
    #     self._simulation_data = value


    @property
    def metric(self):

        '''
        Porperty getter of metric

        Arguments:

            None

        Returns:

            metric
        '''

        return self._metric


    @metric.setter
    def metric(self, value):

        '''
        Property setter for metric

        Arguments:

            value: str, one of euclidean and dtw

        Returns:

            None
        '''

        if not (value == 'euclidean' or value == 'dtw'): 
            raise ValueError(
                f"The metric must be one of euclidean or dtw, but {value} is provided"
            )
        
        self._metric = value


    @property
    def num_clusters(self):

        '''
        Property getter of num_clusters

        Arguments:

            None

        Returns:

            int: number of clusters for the clustering
            (k-means need given number of clusters)
        '''

        return self._num_clusters

    
    @num_clusters.setter
    def num_clusters(self, value):

        '''
        Property setter of num_clusters

        Arguments:
        
            value: int, number of intended clusters
        
        Returns:
        
            None
        '''

        if not isinstance(value, int):
            raise TypeError(
                f"Number of clusters must be integer, but {type(value)} is given"
            )

        self._num_clusters = value


    @property
    def filter_opt(self):

        '''
        Property getter of filter_opt

        Arguments:
        
            None

        Return:
        
            bool: if want filter 0/1 days in clustering
        '''
        return self._filter_opt


    @filter_opt.setter
    def filter_opt(self, value):

        '''
        Property setter of filter_opt

        Arguments:
        
            value: bool.
        
        Returns:
        
            None
        '''

        if not isinstance(value, bool):
            raise TypeError(
                f"filter_opt must be bool, but {type(value)} is given"
            )

        self._filter_opt = value


    def _transform_data(self):

        '''
        Transform the data to clustering package required form.

        Arguments:

            None

        Returns:
            
            train_data: training data for clustering
        '''

        # sclae the data to the capacity factor
        scaled_dispatch_dict_gen = self.simulation_data_gen._scale_data_generator()
        scaled_dispatch_dict_storage = self.simulation_data_storage._scale_data_storage()
        scaled_dispatch_dict = self.simulation_data_gen._scale_data(scaled_dispatch_dict_gen, scaled_dispatch_dict_storage)

        # get the run indexes
        index_list = list(scaled_dispatch_dict.keys())

        # in each simulation data, count 0/1 days.
        if self.filter_opt == True:
            full_day = 0
            zero_day = 0
            day_dataset = []    # slice the annual data into days and put them together

            for idx in index_list:
                sim_year_data = scaled_dispatch_dict[idx]    # sim_year_data is an annual simulation data, 366*24 hours
                day_num = int(len(sim_year_data)/self._time_length)    # calculate the number of days in this annual simulation.
                
                for day in range(day_num):
                    sim_day_data = sim_year_data[day*24:(day+1)*24]    # slice the data into day data with length 24.
                    
                    if sum(sim_day_data) == 0:
                        # it the sum of capacity factor == 0, add a zero day
                        zero_day += 1
                    
                    elif sum(sim_day_data) == 24:
                        # it the sum of capacity factor == 24, add a full day
                        full_day += 1
                    
                    else:
                        day_dataset.append(sim_day_data)
                train_data_year_1 = to_time_series_dataset(day_dataset)

            # use to_time_series_dataset from tslearn to transform the data to the required structure.
            train_data = to_time_series_dataset(day_dataset)
            print(zero_day, full_day)
            return train_data

        # if there is not filter, do not count 0/1 days
        elif self.filter_opt == False:
            day_dataset = []

            for idx in index_list:
                sim_year_data = scaled_dispatch_dict[idx]
                day_num = int(len(sim_year_data)/self._time_length)
                
                for day in range(day_num):
                    sim_day_data = sim_year_data[day*24:(day+1)*24]
                    day_dataset.append(sim_day_data)
            
            return day_dataset


    def _separate_data(self, day_dataset):
        '''
        separate the day data for FE into 'use storage'/'do not use storage'
        '''

        day_dataset_use = []
        day_dataset_not_use = []

        for day in day_dataset:
            c = 0
            for t in day:
                if t > 1:
                    day_dataset_use.append(day)
                    break
                c += 1
            if c == 24:
                day_dataset_not_use.append(day)

        return day_dataset_use,day_dataset_not_use


    def _divide_data(self, day_dataset):
        '''
        Divide data into 2D (thermal generator + stoage)
        '''
        day_dataset_2D = []
        for day in day_dataset:
            storage = []
            for i, t in enumerate(day):
                if t > 1:
                    # unscale the storage capacity into [0,1]
                    storage.append((t - 1)/0.2)
                    day[i] = 1
                else:
                    storage.append(0)
            day_dataset_2D.append([day,storage])
        train_data = to_time_series_dataset(day_dataset_2D)
        
        return train_data


    def clustering_data_kmeans(self):

        '''
        Time series clustering for the dispatch data use kmeans algorithm

        Arguments:

           None

        Returns:
            clustering_model: trained clustering model
        '''

        day_dataset = self._transform_data()

        # for do two parallel clustering method
        # day_dataset_use,day_dataset_not_use = self._separate_data(day_dataset)
        # # use to_time_series_datasets to reshape the data for clustering
        # train_data_use = to_time_series_dataset(day_dataset_use)
        # train_data_not_use = to_time_series_dataset(day_dataset_not_use)

        # clustering_model_use = TimeSeriesKMeans(n_clusters = 5, metric = self.metric, random_state = 0)
        # # model.fit_predict() can fit k-means clustering using X and then predict the closest cluster each time series in X belongs to.
        # clustering_model_use.fit_predict(train_data_use)

        # clustering_model_not_use = TimeSeriesKMeans(n_clusters = 15, metric = self.metric, random_state = 0)
        # clustering_model_not_use.fit_predict(train_data_not_use)

        # for 2D clustering
        train_data = self._transform_data()
        clustering_model = TimeSeriesKMeans(n_clusters = self.num_clusters, metric = self.metric, random_state = 0)
        clustering_model.fit_predict(train_data)

        return clustering_model


    def clustering_data_kmedoids(self):

        '''
        Time series clustering for the dispatch data use kmedoids algorithm

        Arguments:

           None

        Returns:
            clustering_model: trained clustering model
        '''

        day_dataset = self._transform_data()
        # use to_time_series_datasets to reshape the data for clustering
        train_data = to_sklearn_dataset(day_dataset)

        clustering_model = KMedoids(n_clusters=self.num_clusters, init = 'random', random_state=0)
        clustering_model.fit(train_data)

        return clustering_model


    def save_clustering_model(self, clustering_model, fpath = None):

        '''
        Save the model in .json file. fpath can be specified by the user. 

        Arguments:

            clustering_model: trained model from self.clustering_data()

            fpath: if None, save to default path

        Return:

            result_path: result path for the json file. 
        '''

        if fpath == None:    # if none, save to the dafault path
            current_path = os.getcwd()
            result_path =  os.path.join(current_path, f'default_result_path/clustering_result/{self.simulation_data.case_type}_result_{self.simulation_data.num_sims}years_{self.num_clusters}clusters_OD.json')
            clustering_model.to_json(result_path)

        else:    # save to the given path
            if os.path.isabs(fpath) == True:    # if the path is the absolute path
                result_path = fpath
                clustering_model.to_json(result_path)
            else:
                current_path = os.getcwd()
                result_path = os.path.join(current_path,fpath)    # make the path a absolute path
                clustering_model.to_json(result_path)

        return result_path


    def plot_results_kmedoid(self, clustering_model, idx):
        day_dataset = self._transform_data()
        train_data = to_sklearn_dataset(day_dataset)
        centers_dict = {}
        for i, cen in enumerate(clustering_model.cluster_centers_):
            centers_dict[i] = cen

        label_data_dict = {}
        for i, lb in enumerate(clustering_model.labels_):
            if lb not in label_data_dict:
                label_data_dict[lb] = []
                label_data_dict[lb].append(train_data[i])

            else:
                label_data_dict[lb].append(train_data[i])

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
        figname = f'FE_case_study/kmedoid_clustering_figures/NE_kmedoids_result_{self.num_clusters}clusters_cluster{idx}.jpg'
        plt.savefig(figname, dpi = 300)

        return


    def get_cluster_centers(self, result_path):

        '''
        Get the cluster centers from saved file.

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


    def _summarize_results(self, result_path):

        '''
        Summarize the results from the clustering

        Arguments:

            result_path: the path of json file that has clustering results
        
        Returns:

            label_data_dict: dictionary that has the label data {cluster_number:[train_data]}

        '''
        day_dataset = self._transform_data()
        train_data = to_time_series_dataset(day_dataset)

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

    
    def _summarize_results_separate(self, result_path):
        '''
        summarize the results for separate generator and storage clustering problem.

        We need to find the 95/media/5 quantile of summed (generator cf + storage_cf*0.2) data. 

        Then find the generator cf and storage cf.
        '''
        day_dataset = self._transform_data()
        train_data = to_time_series_dataset(day_dataset)

        with open(result_path, 'r') as f:
            cluster_results = json.load(f)
        
        labels = cluster_results['model_params']['labels_']
        
        label_data_dict_index = {}
        label_data_dict = {}
        for idx,lb in enumerate(labels):
            if lb not in label_data_dict:
                label_data_dict[lb] = []
                label_data_dict[lb].append(train_data[idx])
                
                label_data_dict_index[lb] = []
                label_data_dict_index[lb].append(idx)
            else:
                label_data_dict[lb].append(train_data[idx])
                label_data_dict_index[lb].append(idx)
        
        return label_data_dict, label_data_dict_index


    def check_results(self, result_path_use, result_path_not_use):
        '''
        check the mean of all points in the cluster and the cluster center are the same.
        '''
        label_data_dict_use, label_data_dict_not_use = self._summarize_results(result_path_use, result_path_not_use)
        centers_dict_use = self.get_cluster_centers(result_path_use)
        centers_dict_not_use = self.get_cluster_centers(result_path_not_use)

        print(centers_dict_use[1][2])
        cluster0 = []
        for i in label_data_dict_use[1]:
            cluster0.append(i[2])
        print(len(cluster0))
        print(np.mean(cluster0))



    def plot_results_2D(self,result_path, fpath = None):
        current_path = os.getcwd()
        label_data_dict = self._summarize_results_2D(result_path)
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
                ax2.plot(time_length, data[1], '--', c='g', alpha=0.05)

            ax1.plot(time_length, centers_dict[idx][0], '-', c='r', alpha=1.0)
            ax2.plot(time_length, centers_dict[idx][1], '-', c='r', alpha=1.0)
            ax1.set_ylabel('Capacity factor',font = font1)
            ax2.set_ylabel('Capacity factor',font = font1)
            ax1.set_xlabel('Time(h)',font = font1)
            ax2.set_xlabel('Time(h)',font = font1)
            figname = os.path.join(f'{self.simulation_data.case_type}_case_study','clustering_figures',f'{self.simulation_data.case_type}_result_2D_cluster{idx}.jpg')
            plt.savefig(figname, dpi = 300)

        return  


    def plot_centers(self, result_path, fpath = None):
        
        '''
        plot the representative days in one plot

        Arguments:
            
            result_path: the path of json file that has clustering results

        Returns:

            None
        '''
        print('Making center plots')
        time_length = range(24)
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }      

        centers_dict = self.get_cluster_centers(result_path)
        f,ax = plt.subplots(figsize = ((16,6)))
        for key, value in centers_dict.items():
            ax.plot(time_length, value, label = f'cluster_{key}')

        ax.set_xlabel('Time(h)',font = font1)
        ax.set_ylabel('Capacity factor',font = font1)
        if fpath == None:
            figname = f'{self.simulation_data.case_type}_case_study/clustering_figures/{self.simulation_data.case_type}_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_whole_centers.jpg'
        else:
            # if the path is given, save to it. 
            figname = fpath          
        plt.savefig(figname, dpi = 300)


    def _find_max_min_mileage(self, result_path):
        '''
        Find within the cluster, which data point has the max mileage.

        Arguments:
    
            result_path: the path of json file that has clustering results

        Returns:

            mileage: dict, keys: cluster number, values: list, mileage of each data point, 
        '''

        label_data_dict = self._summarize_results(result_path)

        mileage = {}
        # for every cluster
        for idx in label_data_dict:
            # for every data points in this cluster
            mileage[idx] = []
            for d in range(len(label_data_dict[idx])):
                # flatten the 2D list (from to_time_series_dataset) to 1D array
                d_array = np.array(label_data_dict[idx][d]).flatten()
                # calculate the mileage and save them in the list
                abs_milage = np.abs(np.diff(d_array))
                mileage[idx].append(np.sum(abs_milage))

        return mileage

    def box_plots(self, result_path, fpath=None):
        
        '''
        Generate box plots for analyzing the clustering resuls.

        Arguments: 

            result_path: the path of json file that has clustering results

        return:
            
            outlier_count: dict, count the number of outliers in each cluster.

        '''


        # read the cluster centers in numpy.ndarray
        print('Making box plots')

        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

        cluster_max_dispatch, cluster_95_dispatch, cluster_median_dispatch, cluster_5_dispatch, cluster_min_dispatch \
         = self.find_dispatch_max_min(result_path)

        with open(result_path, 'r') as f:
            cluster_results = json.load(f)

        # read cluster centers
        centers = np.array(cluster_results['model_params']['cluster_centers_'])

        # read the label results
        res_dict = self._summarize_results(result_path)

        # 5 clusters in one plot
        if self.num_clusters%5 >= 1:
            plot_num = self.num_clusters//5 +1
        else:
            plot_num = self.num_clusters//5

        # read mileage data
        mileage = self._find_max_min_mileage(result_path)

        p = 1
        outlier_count = {}
        while p <= plot_num - 1:
            fig_res_list = []
            fig_label = []
            cf_center = []
            max_mileage = []
            min_mileage = []
            q_95 = []
            q_5 = []
            max_cf = []
            min_cf = []
            for i in range((p-1)*5,p*5):
                res_array = np.array(res_dict[i])
                res_list = []
                #calculate the capacity factor
                for j in res_array:
                    res_list.append(sum(j)/24)
                fig_res_list.append(np.array(res_list).flatten())
                # calculate the percentage of points in the cluster
                percentage = np.round(len(res_array)/self.simulation_data.num_sims/364*100,2)
                cf_center.append(sum(centers[i])/24)
                # count the outliers
                Q1 = np.quantile(np.array(res_list).flatten(), 0.25)
                Q3 = np.quantile(np.array(res_list).flatten(), 0.75)
                gap = 1.5*(Q3-Q1)
                lower = np.sum(np.array(res_list).flatten()<= Q1-gap)
                higher = np.sum(np.array(res_list).flatten()>= Q3+gap)
                outlier_count[i] = np.round((lower+higher)/len(np.array(res_list).flatten())*100,4)
                # make label
                fig_label.append(f'cluster_{i}'+'\n'+str(percentage)+'%'+'\n'+str(outlier_count[i])+'%')
                # max/min mileage
                max_mileage.append(np.array([np.sum(res_array[np.argmax(mileage[i])])/24]))
                min_mileage.append(np.array([np.sum(res_array[np.argmin(mileage[i])])/24]))
                # 95% and 5% quantile
                q_95.append(np.array([np.sum(cluster_95_dispatch[i])/24]))
                q_5.append(np.array([np.sum(cluster_5_dispatch[i])/24]))
                # max / min
                max_cf.append(np.array([np.max(res_list)]))
                min_cf.append(np.array([np.min(res_list)]))

            f,ax = plt.subplots(figsize = (8,6))
            ax.boxplot(fig_res_list,labels = fig_label, medianprops = {'color':'k'})
            ax.boxplot(cf_center, labels = fig_label,medianprops = {'color':'r'})
            ax.boxplot(max_mileage, labels = fig_label, medianprops = {'color':'c'})
            ax.boxplot(min_mileage, labels = fig_label, medianprops = {'color':'y'})
            ax.boxplot(q_95, labels = fig_label, medianprops = {'color':'brown'})
            ax.boxplot(q_5, labels = fig_label, medianprops = {'color':'pink'})
            ax.boxplot(max_cf, labels = fig_label, medianprops = {'color':'b'})
            ax.boxplot(min_cf, labels = fig_label, medianprops = {'color':'m'})
            ax.set_ylabel('capacity_factor', font = font1)
            figname = os.path.join(f"{self.simulation_data.case_type}_case_study","clustering_figures",f"{self.simulation_data.case_type}_box_plot_{self.num_clusters}clusters_{p}.jpg")
            # plt.savefig will not overwrite the existing file
            plt.savefig(figname,dpi =300)
            p += 1


        fig_res_list = []
        fig_label = []
        cf_center = []
        max_mileage = []
        min_mileage = []
        q_95 = []
        q_5 = []
        max_cf = []
        min_cf = []
        for i in range((plot_num-1)*5, self.num_clusters):
            res_array = np.array(res_dict[i])
            res_list = []
            #calculate the capacity factor
            for j in res_array:
                res_list.append(sum(j)/24)
            fig_res_list.append(np.array(res_list).flatten())
            percentage = np.round(len(res_array)/self.simulation_data.num_sims/364*100,2)
            cf_center.append(sum(centers[i])/24)
            Q1 = np.quantile(np.array(res_list).flatten(), 0.25)
            Q3 = np.quantile(np.array(res_list).flatten(), 0.75)
            gap = 1.5*(Q3-Q1)
            lower = np.sum(np.array(res_list).flatten()<= Q1-gap)
            higher = np.sum(np.array(res_list).flatten()>= Q3+gap)
            outlier_count[i] = np.round((lower+higher)/len(np.array(res_list).flatten())*100,4)
            fig_label.append(f'cluster_{i}'+'\n'+str(percentage)+'%'+'\n'+str(outlier_count[i])+'%')
            max_mileage.append(np.array([np.sum(res_array[np.argmax(mileage[i])])/24]))
            min_mileage.append(np.array([np.sum(res_array[np.argmin(mileage[i])])/24]))
            # 95% and 5% quantile
            q_95.append(np.array([np.sum(cluster_95_dispatch[i])/24]))
            q_5.append(np.array([np.sum(cluster_5_dispatch[i])/24]))
            # max / min
            max_cf.append(np.array([np.max(res_list)]))
            min_cf.append(np.array([np.min(res_list)]))
        f,ax = plt.subplots(figsize = (8,6))
        ax.boxplot(fig_res_list,labels = fig_label, medianprops = {'color':'k'})
        ax.boxplot(cf_center, labels = fig_label,medianprops = {'color':'r'})
        ax.boxplot(max_mileage, labels = fig_label, medianprops = {'color':'c'})
        ax.boxplot(min_mileage, labels = fig_label, medianprops = {'color':'y'})
        ax.boxplot(q_95, labels = fig_label, medianprops = {'color':'brown'})
        ax.boxplot(q_5, labels = fig_label, medianprops = {'color':'pink'})
        ax.boxplot(max_cf, labels = fig_label, medianprops = {'color':'b'})
        ax.boxplot(min_cf, labels = fig_label, medianprops = {'color':'m'})
        ax.set_ylabel('capacity_factor', font = font1)
        
        if fpath == None:
            figname = os.path.join(f"{self.simulation_data.case_type}_case_study","clustering_figures",f"{self.simulation_data.case_type}_box_plot_{self.num_clusters}clusters_{p}.jpg")
        else:
            # if the path is given, save to it. 
            figname = fpath
        plt.savefig(figname,dpi =300)
        
        return outlier_count


    def find_dispatch_max_min(self, result_path):
        '''
        Find the max and min wind profile within the cluster.  
        '''
        label_data_dict, label_data_dict_index = self._summarize_results_separate(result_path)
        centers_dict = self.get_cluster_centers(result_path)
        
        font1 = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 15,
        }

        # mileage = self._find_max_min_mileage(result_path)
        time_length = range(24)
        # cluster_max_dispatch = {}
        # cluster_min_dispatch = {}
        cluster_95_dispatch = {}
        cluster_5_dispatch = {}
        cluster_median_dispatch = {}
        cluster_95_dispatch_index = {}
        cluster_5_dispatch_index = {}
        cluster_median_dispatch_index = {}

        for idx in range(self.num_clusters):
            sum_dispatch_data = []
            for data in label_data_dict[idx]:
                sum_dispatch_data.append(np.sum(data))
            # 95% and 5% qunatile index

            # cluster_max_dispatch[idx] = label_data_dict[idx][np.argmax(sum_dispatch_data)].tolist()
            # cluster_min_dispatch[idx] = label_data_dict[idx][np.argmin(sum_dispatch_data)].tolist()
            median_index = np.argsort(sum_dispatch_data)[len(sum_dispatch_data) // 2]
            quantile_95_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.95)]
            quantile_5_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.05)]

            # record the dispatch data of median/95/5
            cluster_95_dispatch[idx] = label_data_dict[idx][quantile_95_index].tolist()
            cluster_5_dispatch[idx] = label_data_dict[idx][quantile_5_index].tolist()
            cluster_median_dispatch[idx] = label_data_dict[idx][median_index].tolist()
            # record the index of the median/95/5
            cluster_95_dispatch_index[idx] = label_data_dict_index[idx][quantile_95_index]
            cluster_5_dispatch_index[idx] = label_data_dict_index[idx][quantile_5_index]
            cluster_median_dispatch_index[idx] = label_data_dict_index[idx][median_index]

        with open('FE_dispatch_95_5_median_new.json', 'w') as f:
            json.dump({'cluster_95_dispatch':cluster_95_dispatch, 'cluster_5_dispatch': cluster_5_dispatch, 'median_dispatch':cluster_median_dispatch}, f)

        # for idx in range(self.num_clusters):
        #     f,ax = plt.subplots()
        #     for data in label_data_dict[idx]:
        #         ax.plot(time_length, data, '--', c='g', alpha=0.05)
        #     cf_center = np.sum(centers_dict[idx])/24
        #     ax.plot(time_length, centers_dict[idx], '-', c='r', linewidth=3, alpha=1.0, label = f'representative ({round(cf_center,3)})')
        #     cf_95 = np.sum(cluster_95_dispatch[idx])/24
        #     ax.plot(time_length, cluster_95_dispatch[idx], '-', c='brown', linewidth=3, alpha=1.0, label = f'95 quantile ({round(cf_95,3)})')
        #     cf_5 = np.sum(cluster_5_dispatch[idx])/24
        #     ax.plot(time_length, cluster_5_dispatch[idx], '-', c='pink', linewidth=3, alpha=1.0, label = f'5 quantile ({round(cf_5,3)})')
        #     cf_med = np.sum(cluster_median_dispatch[idx])/24
        #     ax.plot(time_length, cluster_median_dispatch[idx], '-', c='k', linewidth=3, alpha=1.0, label = f'median ({round(cf_med,3)})')
        #     # cf_max = np.sum(cluster_max_dispatch[idx])/24
        #     # ax.plot(time_length, cluster_max_dispatch[idx], '-', c='b', linewidth=3, alpha=1.0, label = f'max ({round(cf_max,3)})')
        #     # cf_min = np.sum(cluster_min_dispatch[idx])/24
        #     # ax.plot(time_length, cluster_min_dispatch[idx], '-', c='m', linewidth=3, alpha=1.0, label = f'min ({round(cf_min,3)})')
        #     # cf_max_ramp = np.sum(label_data_dict[idx][np.argmax(mileage[idx])])/24
        #     # ax.plot(time_length, label_data_dict[idx][np.argmax(mileage[idx])], '-', c='c', linewidth=3, alpha=1.0, label = f'max_mileage ({round(cf_max_ramp,3)})')
        #     # cf_min_ramp = np.sum(label_data_dict[idx][np.argmin(mileage[idx])])/24
        #     # ax.plot(time_length, label_data_dict[idx][np.argmin(mileage[idx])], '-', c='y', linewidth=3, alpha=1.0, label = f'min_mileage ({round(cf_min_ramp,3)})')
        #     ax.tick_params(direction = 'in')
        #     ax.set_title(f'cluster_{idx}')
        #     ax.set_ylabel('Capacity factor',font = font1)
        #     ax.set_xlabel('Time(h)',font = font1)
        #     ax.legend()
        #     figname = f'clustering_figures/FE_dispatch_95_5_mileage_{idx}.jpg'
        #     plt.savefig(figname, dpi = 300)

        return cluster_95_dispatch_index, cluster_5_dispatch_index, cluster_median_dispatch_index


    def find_target_gen_storage_data(self, cluster_95_dispatch_index, cluster_5_dispatch_index, cluster_median_dispatch_index):
        '''
        Find the target data 
        '''
        scaled_dispatch_dict_gen = self.simulation_data_gen._scale_data_generator()
        scaled_dispatch_dict_storage = self.simulation_data_storage._scale_data_storage()
        scaled_dispatch_dict = self.simulation_data_gen._scale_data(scaled_dispatch_dict_gen, scaled_dispatch_dict_storage)

        # get the run indexes
        index_list = list(scaled_dispatch_dict.keys())
        if self.filter_opt == True:
            full_day = 0
            zero_day = 0
            day_dataset = []    # slice the annual data into days and put them together
            separate_dataset_gen = []
            separate_dataset_storage = []

            for idx in index_list:
                sim_year_data = scaled_dispatch_dict[idx]    # sim_year_data is an annual simulation data, 366*24 hours
                day_num = int(len(sim_year_data)/self._time_length)    # calculate the number of days in this annual simulation.
                sim_year_data_gen = scaled_dispatch_dict_gen[idx]
                sim_year_data_storage = scaled_dispatch_dict_storage[idx]

                for day in range(day_num):
                    sim_day_data = sim_year_data[day*24:(day+1)*24]    # slice the data into day data with length 24.
                    sim_day_data_gen = sim_year_data_gen[day*24:(day+1)*24]
                    sim_day_data_storage = sim_year_data_storage[day*24:(day+1)*24]

                    if sum(sim_day_data) == 0:
                        # it the sum of capacity factor == 0, add a zero day
                        zero_day += 1
                    
                    elif sum(sim_day_data) == 24:
                        # it the sum of capacity factor == 24, add a full day
                        full_day += 1
                    
                    else:
                        day_dataset.append(sim_day_data)
                        separate_dataset_gen.append(sim_day_data_gen)
                        separate_dataset_storage.append(sim_day_data_storage)

            train_data_gen = to_time_series_dataset(separate_dataset_gen)
            train_data_storage = to_time_series_dataset(separate_dataset_storage)
            
            with open('FE_dispatch_95_5_median_new.json', 'rb') as f:
                dt = json.load(f)
            
            cluster_95_dispatch_gen = {}
            cluster_5_dispatch_gen = {}
            cluster_median_dispatch_gen = {}
            cluster_95_dispatch_storage = {}
            cluster_5_dispatch_storage = {}
            cluster_median_dispatch_storage = {}

            for i in range(self.num_clusters):
                index_5 = cluster_5_dispatch_index[i]
                index_95 = cluster_95_dispatch_index[i]
                index_med = cluster_median_dispatch_index[i]

                gen_data_95 = train_data_gen[index_95]
                gen_data_5 = train_data_gen[index_5]
                gen_data_med = train_data_gen[index_med]
                storage_data_95 = train_data_storage[index_95]
                storage_data_5 = train_data_storage[index_5]
                storage_data_med = train_data_storage[index_med]
                # ndarray is not json serializable
                cluster_95_dispatch_gen[i] = gen_data_95.tolist()
                cluster_5_dispatch_gen[i] = gen_data_5.tolist()
                cluster_median_dispatch_gen[i] = gen_data_med.tolist()
                cluster_95_dispatch_storage[i] = storage_data_95.tolist()
                cluster_5_dispatch_storage[i] = storage_data_5.tolist()
                cluster_median_dispatch_storage[i] = storage_data_med.tolist()
                
                if not np.allclose(gen_data_5 + storage_data_5*0.2,dt['cluster_5_dispatch'][str(i)]):
                    print('5', i)
                if not np.allclose(gen_data_95 + storage_data_95*0.2,dt['cluster_95_dispatch'][str(i)]):
                    print('95', i)
                if not np.allclose(gen_data_med + storage_data_med*0.2,dt['median_dispatch'][str(i)]):
                    print('median', i)

            with open ('FE_dispatch_separate.json','w') as f:
                json.dump({'cluster_95_dispatch_gen':cluster_95_dispatch_gen, 'cluster_5_dispatch_gen':cluster_5_dispatch_gen, 'cluster_median_dispatch_gen':cluster_median_dispatch_gen,
                            'storage_data_95':cluster_95_dispatch_storage, 'storage_data_5':cluster_5_dispatch_storage, 'storage_data_med':cluster_median_dispatch_storage}, f)
        return
    # def cluster_analysis(self, result_path):

    #     label_data_dict = self._summarize_results_1D(result_path)
    #     centers_dict = self.get_cluster_centers(result_path)

    #     for idx in range(self.num_clusters):
    #         sum_dispatch_capacity_factor = []
    #         for data in label_data_dict[idx]:
    #             sum_dispatch_capacity_factor.append(np.sum(data)/24*10)
    #         fig,ax = plt.subplots()
    #         bins = list(range(11))
    #         ax.hist(sum_dispatch_capacity_factor, bins = bins, density = True, label = 'Dispatch')
    #         ax.set_ylabel('Probability Density')
    #         ax.set_xlabel('Day Capacity Factors')
    #         ax.set_title(f'Dispatch histogram cluster_{idx}')
    #         ax.legend()
    #         plt.savefig(f'histogram_cluster_{idx}.jpg')
