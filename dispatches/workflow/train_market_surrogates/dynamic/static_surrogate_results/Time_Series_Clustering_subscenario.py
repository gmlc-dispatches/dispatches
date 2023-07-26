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
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn_extra.cluster import KMedoids
from tslearn.utils import to_sklearn_dataset
import numpy as np
import json
import matplotlib.pyplot as plt

# this script supports subscenario anaylsis.
class TimeSeriesClustering:

    def __init__(self, num_clusters, simulation_data, filter_opt = True, metric = 'euclidean'):
        
        ''' 
        Time series clustering for the dispatch data. 

        Now only can do clustering over dispatch data.

        Arguments:
            
            simulation data: object, composition from ReadData class
            
            metric: metric for clustering, must be one of euclidean or dtw

            num_clusters: number of clusters that we want to have

            filter_opt: bool, if we are going to filter out 0/1 capacity days
        
        '''
        self.simulation_data = simulation_data
        self.metric = metric
        self.num_clusters = num_clusters
        self.filter_opt = filter_opt
        # set a class property which is the time length of a day.
        self._time_length = 24


    @property
    def simulation_data(self):

        '''
        Porperty getter of simulation_data
        
        Arguments:

            None

        Returns:
            simulation_data
        '''
        
        return self._simulation_data


    @simulation_data.setter
    def simulation_data(self, value):

        '''
        Porperty setter of simulation_data
        
        Arguments:

            value: object, composition from ReadData class

        Returns:
            None
        '''
        
        if not isinstance(value, object):
            raise TypeError(
                f"The simulation_data must be an object, but {type(value)} is given."
            )
        self._simulation_data = value


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
        scaled_dispatch_dict = self.simulation_data._scale_data()

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


    def clustering_data_kmeans(self):

        '''
        Time series clustering for the dispatch data use kmeans algorithm

        Arguments:

           None

        Returns:
            clustering_model: trained clustering model
        '''

        day_dataset = self._transform_data()
        # use to_time_series_datasets to reshape the data for clustering
        train_data = to_time_series_dataset(day_dataset)
        
        clustering_model = TimeSeriesKMeans(n_clusters = self.num_clusters, metric = self.metric, random_state = 0)
        # model.fit_predict() can fit k-means clustering using X and then predict the closest cluster each time series in X belongs to.
        labels = clustering_model.fit_predict(train_data)

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

        train_data = self._transform_data()

        with open(result_path, 'r') as f:
            cluster_results = json.load(f)
        
        # load the label data
        labels = cluster_results['model_params']['labels_']

        # make the result a dictionary {label: [data_1, data_2,...,}
        label_data_dict = {}
        for idx,lb in enumerate(labels):
            
            if lb not in label_data_dict:
                label_data_dict[lb] = []
                label_data_dict[lb].append(train_data[idx])
            
            else:
                label_data_dict[lb].append(train_data[idx])

        return label_data_dict


    def plot_results(self, result_path, idx, fpath = None):
        
        '''
        Plot the result data. Each plot is the represenatative days and data in the cluster.

        Arguments: 

            result_path: the path of json file that has clustering results

            idx: int, the index that of the cluster center

            fpath: the path to save the plot

        Returns:

            None
        '''

        # print('Making clustering plots')

        label_data_dict = self._summarize_results(result_path)
        centers_dict = self.get_cluster_centers(result_path)

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
        if fpath == None:
            figname = f'{self.simulation_data.case_type}_case_study/clustering_figures/{self.simulation_data.case_type}_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_cluster{idx}.jpg'
        else:
            # if the path is given, save to it. 
            figname = fpath
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
        with open(result_path, 'r') as f:
            cluster_results = json.load(f)

        centers = np.array(cluster_results['model_params']['cluster_centers_'])

        # read the label results
        res_dict = self._summarize_results(result_path)

        # 5 clusters in one plot
        if self.num_clusters%5 >= 1:
            plot_num = self.num_clusters//5 +1
        else:
            plot_num = self.num_clusters//5

        p = 1
        outlier_count = {}
        while p <= plot_num - 1:
            fig_res_list = []
            fig_label = []
            cf_center = []
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
                fig_label.append(f'cluster_{i}'+'\n'+str(percentage)+'%'+'\n'+str(outlier_count[i])+'%')

            f,ax = plt.subplots(figsize = (8,6))
            ax.boxplot(fig_res_list,labels = fig_label, medianprops = {'color':'g'})
            ax.boxplot(cf_center, labels = fig_label,medianprops = {'color':'r'})
            ax.set_ylabel('capacity_factor', font = font1)
            figname = os.path.join(f"{self.simulation_data.case_type}_case_study","clustering_figures",f"{self.simulation_data.case_type}_box_plot_{self.num_clusters}clusters_{p}.jpg")
            # plt.savefig will not overwrite the existing file
            plt.savefig(figname,dpi =300)
            p += 1


        fig_res_list = []
        fig_label = []
        cf_center = []
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
        f,ax = plt.subplots(figsize = (8,6))
        ax.boxplot(fig_res_list,labels = fig_label, medianprops = {'color':'g'})
        ax.boxplot(cf_center, labels = fig_label,medianprops = {'color':'r'})
        ax.set_ylabel('capacity_factor', font = font1)
        
        if fpath == None:
            figname = os.path.join(f"{self.simulation_data.case_type}_case_study","clustering_figures",f"{self.simulation_data.case_type}_box_plot_{self.num_clusters}clusters_{p}.jpg")
        else:
            # if the path is given, save to it. 
            figname = fpath
        plt.savefig(figname,dpi =300)
        
        return outlier_count