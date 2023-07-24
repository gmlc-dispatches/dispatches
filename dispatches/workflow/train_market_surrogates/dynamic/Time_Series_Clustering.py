#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

import json
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData

plt.rcParams["figure.figsize"] = (12,9)


class TimeSeriesClustering:

    def __init__(self, simulation_data, num_clusters, filter_opt = True, metric = 'euclidean'):
        
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
        # the case type is inherited from the self.simulation_data 
        self.case_type = self.simulation_data.case_type
        
        # for RE, the filter option cannot be True
        if self.case_type == 'RE' and self.filter_opt == True:
            raise TypeError(
                'f{self.case_type} cannot have set the filter_opt to \'True\'. '
            )



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

        if not isinstance(value, SimulationData):
            raise TypeError(
                f"The simulation_data must be created from SimulationData."
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
                f"Filter_opt must be bool, but {type(value)} is given"
            )

        self._filter_opt = value


    def _transform_data_RE(self):

        '''
        shape the data to the format that tslearn can read. 
        
        This is for RE case study because we do 2d clustering in RE. 

        Arguments:
            dispatch data in the shape of numpy array. (Can be obtained from self.read_data())

        Return:
            train_data: np.arrya for the tslearn package. Dimension = (self.years*364, 24, 1)
            data of full/zero days: [zero_day,full_day]
        '''
        # read the wind data 
        wind_data = self.simulation_data.read_wind_data()
        
        # number of hours in a representative day
        time_len = 24
        
        # sclae the data to the capacity factor
        scaled_dispatch_dict = self.simulation_data._scale_data()

        # get the run indexes
        index_list = list(scaled_dispatch_dict.keys())

        full_day = []
        zero_day = []
        day_dataset = []

        for idx in index_list:
            # slice the year data into day data(24 hours a day)
            sim_year_data = scaled_dispatch_dict[idx]
            day_num = int(len(sim_year_data)/time_len)

            if self.filter_opt == True:
            # filter out full/zero capacity days
                for i in range(day_num):
                    dispatch_day_data = sim_year_data[i*time_len:(i+1)*time_len]
                    # count the day of full/zero capacity factor.
                    # Sepearte the data out. np.shape(zero/full_day) = (num_days, 2, 24)
                    if sum(dispatch_day_data) == 0:
                        zero_day.append([dispatch_day_data,wind_data[i]])
                    elif sum(dispatch_day_data) == 24:
                        full_day.append([dispatch_day_data,wind_data[i]])
                    else:
                        # np.shape(datasets) = (num_days, 2, 24)
                        # (wind(1*24), dispatch(1*24))
                        day_dataset.append([dispatch_day_data,wind_data[i]])
            # no filter
            else:
                for i in range(day_num):
                    dispatch_day_data = sim_year_data[i*time_len:(i+1)*time_len]
                    day_dataset.append([dispatch_day_data,wind_data[i]])
        
        # use tslearn package to form the correct data structure.
        train_data = to_time_series_dataset(day_dataset)

        return train_data


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

        # For RE, we do 2D clustering for wind and dispatch
        if self.case_type == 'RE':
            train_data = self._transform_data_RE()
            
            return train_data

        else:
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
                # print the zero and full days
                print(f'The number of zero capacity days in the dataset is {zero_day}')
                print(f'The number of full capacity days in the dataset is {full_day}')
                
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
                
                print('No filter')

                train_data = to_time_series_dataset(day_dataset)
                
                return train_data


    def clustering_data(self):

        '''
        Time series clustering for the dispatch data

        Arguments:

           None

        Returns:
            clustering_model: trained clustering model
        '''

        train_data = self._transform_data()

        clustering_model = TimeSeriesKMeans(n_clusters = self.num_clusters, metric = self.metric, random_state = 42)
        # model.fit_predict() can fit k-means clustering using X and then predict the closest cluster each time series in X belongs to.
        labels = clustering_model.fit_predict(train_data)

        return clustering_model


    def save_clustering_model(self, clustering_model, fpath = None):

        '''
        Save the model in .json file. fpath can be specified by the user. 

        Arguments:

            clustering_model: trained model from self.clustering_data()

            fpath: if None, save to current path

        Return:

            result_path: result path for the json file. 
        '''

        if fpath == None:    # if none, save to the current path
            result_path = str(pathlib.Path.cwd().joinpath(f'{self.case_type}_case_study', f'{self.simulation_data.case_type}_result_{self.simulation_data.num_sims}years_{self.num_clusters}clusters_OD.json'))
            clustering_model.to_json(result_path)

        else:    # save to the given path
            result_path = str(pathlib.Path(fpath).absolute())   # make the path a absolute path
            clustering_model.to_json(result_path)

        return result_path


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


    def plot_results(self, result_path):
        
        '''
        Plot the result data. Each plot is the represenatative days and data in the cluster.

        Different case studies needs different data processing.

        Arguments: 

            result_path: the path of json file that has clustering results

        Returns:

            result_list: list of clustering information
        '''
        # the RE is 2D clustering, so there are some difference in plotting
        if self.case_type == 'FE' or self.case_type == 'NE':
            result_list = self.plot_result_NE_FE(result_path)
        
        else:
            result_list = self.plot_result_RE(result_path)

        return result_list


    def plot_result_NE_FE(self, result_path):
        '''
        Find the median, 95% and 5% cf dispatch profile within the cluster. 

        Arguments: 

            result_path: the path of json file that has clustering results
        
        Returns:

            dispatch_result: list, dispaatch_results = [cluster_95_dispatch, cluster_5_dispatch, cluster_median_dispatch]
        '''
        # get label and cluster centers
        label_data_dict = self._summarize_results(result_path)
        centers_dict = self.get_cluster_centers(result_path)
        
        font1 = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 15,
        }

        time_length = range(24)
        # defind 5 dictionaries to store the data.
        cluster_95_dispatch = {}
        cluster_5_dispatch = {}
        cluster_median_dispatch = {}
        for idx in range(self.num_clusters):
            sum_dispatch_data = []
            # sum the 24 hour cf for each day in the cluster.
            for data in label_data_dict[idx]:
                sum_dispatch_data.append(np.sum(data))
            # find out the median, 95% and 5% qunatile index.
            median_index = np.argsort(sum_dispatch_data)[len(sum_dispatch_data) // 2]
            quantile_95_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.95)]
            quantile_5_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.05)]
            # convert the time series data to index.
            cluster_95_dispatch[idx] = label_data_dict[idx][quantile_95_index].tolist()
            cluster_5_dispatch[idx] = label_data_dict[idx][quantile_5_index].tolist()
            cluster_median_dispatch[idx] = label_data_dict[idx][median_index].tolist()

        with open(f'{self.case_type}_case_study/{self.simulation_data.case_type}_dispatch_95_5_median_new.json', 'w') as f:
            json.dump({'cluster_95_dispatch':cluster_95_dispatch, 'cluster_5_dispatch': cluster_5_dispatch, 'median_dispatch':cluster_median_dispatch}, f)

        for idx in range(self.num_clusters):
            f,ax = plt.subplots()
            for data in label_data_dict[idx]:
                ax.plot(time_length, data, '--', c='g', alpha=0.05)
            cf_center = np.sum(centers_dict[idx])/24
            ax.plot(time_length, centers_dict[idx], '-', c='r', linewidth=3, alpha=1.0, label = f'representative ({round(cf_center,3)})')
            cf_95 = np.sum(cluster_95_dispatch[idx])/24
            ax.plot(time_length, cluster_95_dispatch[idx], '-', c='brown', linewidth=3, alpha=1.0, label = f'95 quantile ({round(cf_95,3)})')
            cf_5 = np.sum(cluster_5_dispatch[idx])/24
            ax.plot(time_length, cluster_5_dispatch[idx], '-', c='pink', linewidth=3, alpha=1.0, label = f'5 quantile ({round(cf_5,3)})')
            cf_med = np.sum(cluster_median_dispatch[idx])/24
            ax.plot(time_length, cluster_median_dispatch[idx], '-', c='k', linewidth=3, alpha=1.0, label = f'median ({round(cf_med,3)})')
            ax.tick_params(direction = 'in')
            ax.set_title(f'cluster_{idx}')
            ax.set_ylabel('Capacity factor',font = font1)
            ax.set_xlabel('Time(h)',font = font1)
            ax.legend()

            # save to default path
            folder_path = f'{self.case_type}_case_study/clustering_figures'
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            figname = str(pathlib.Path.cwd().joinpath(folder_path, f'{self.case_type}_dispatch_cluster_{idx}.jpg'))
            plt.savefig(figname, dpi = 300)
        
        dispatch_result = [cluster_95_dispatch, cluster_5_dispatch, cluster_median_dispatch]

        return dispatch_result


    def plot_result_RE(self, result_path):

        '''
        Find the max, min, median, 95% and 5% cf dispatch profile within the cluster.  
        
        Arguments: 

            result_path: the path of json file that has clustering results
        
        Returns:

            combined_results: list, [dispatch_results, wind_results], 
                            where dispaatch_results = [cluster_95_dispatch, cluster_5_dispatch, cluster_median_dispatch]
                            wind_results = [cluster_95_wind, cluster_5_wind, cluster_median_wind]
        '''

        label_data_dict = self._summarize_results(result_path)
        centers_dict = self.get_cluster_centers(result_path)
        
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        time_length = range(24)
        cluster_95_dispatch = {}
        cluster_5_dispatch = {}
        cluster_median_dispatch = {}
        cluster_95_wind = {}
        cluster_5_wind = {}
        cluster_median_wind = {}

        for idx in range(self.num_clusters):
            cluster_95_dispatch[idx] = []
            cluster_5_dispatch[idx] = []
            cluster_median_dispatch[idx] = []
            cluster_95_wind[idx] = []
            cluster_5_wind[idx] = []
            cluster_median_wind[idx] = []
            sum_dispatch_data = []
            for data in label_data_dict[idx]:
                sum_dispatch_data.append(np.sum(data[0]))

            median_index = np.argsort(sum_dispatch_data)[len(sum_dispatch_data) // 2]
            quantile_95_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.95)]
            quantile_5_index = np.argsort(sum_dispatch_data)[int(len(sum_dispatch_data)*0.05)]
            cluster_95_dispatch[idx].append(label_data_dict[idx][quantile_95_index][0].tolist())
            cluster_95_wind[idx].append(label_data_dict[idx][quantile_95_index][1].tolist())
            cluster_5_dispatch[idx].append(label_data_dict[idx][quantile_5_index][0].tolist())
            cluster_5_wind[idx].append(label_data_dict[idx][quantile_5_index][1].tolist())
            cluster_median_dispatch[idx].append(label_data_dict[idx][median_index][0].tolist())
            cluster_median_wind[idx].append(label_data_dict[idx][median_index][1].tolist())

        with open(f'{self.case_type}_case_study/RE_dispatch_95_5_median.json', 'w') as f:
            json.dump({'cluster_95_dispatch':cluster_95_dispatch, 'cluster_5_dispatch': cluster_5_dispatch, 'median_dispatch':cluster_median_dispatch,\
                'cluster_95_wind':cluster_95_wind, 'cluster_5_wind':cluster_5_wind, 'median_wind':cluster_median_wind}, f)

        for idx in range(self.num_clusters):
            f,(ax0,ax1) = plt.subplots(2,1)
            for data in label_data_dict[idx]:
                ax0.plot(time_length, data[0], '--', c='g', alpha=0.05)
                ax1.plot(time_length, data[1], '--', c='g', alpha=0.05)
            ax0.plot(time_length, centers_dict[idx][0], '-', c='r', alpha=1.0, label = 'mean')
            ax1.plot(time_length, centers_dict[idx][1], '-', c='r', alpha=1.0, label = 'mean')
            ax0.plot(time_length, cluster_95_dispatch[idx][0], '-', c='b', alpha=1.0, label = '95 quantile')
            ax1.plot(time_length, cluster_95_wind[idx][0], '-', c='b', alpha=1.0, label = '95 quantile')
            ax0.plot(time_length, cluster_5_dispatch[idx][0], '-', c='k', alpha=1.0, label = '5 quantile')
            ax1.plot(time_length, cluster_5_wind[idx][0], '-', c='k', alpha=1.0, label = '5 quantile')
            ax0.plot(time_length, cluster_median_dispatch[idx][0], '-', c='m', alpha=1.0, label = 'median')
            ax1.plot(time_length, cluster_median_wind[idx][0], '-', c='m', alpha=1.0, label = 'median')
            ax0.set_ylabel('Capacity factor',font = font1)
            ax0.set_xlabel('Time(h)',font = font1)
            ax1.set_ylabel('Capacity factor',font = font1)
            ax1.set_xlabel('Time(h)',font = font1)
            ax0.legend()
            ax1.legend()
            ax0.set_title('Dispatch Profile')
            ax1.set_title('Wind Profile')

            # save to default path
            folder_path = f'{self.case_type}_case_study/clustering_figures'
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            figname = str(pathlib.Path.cwd().joinpath(folder_path, f'{self.case_type}_dispatch_cluster_{idx}.jpg'))
            plt.savefig(figname, dpi = 300)
        
        dispatch_results = [cluster_95_dispatch, cluster_5_dispatch, cluster_median_dispatch]
        wind_results = [cluster_95_wind, cluster_5_wind, cluster_median_wind]
        combined_results = [dispatch_results, wind_results]

        return combined_results


    def plot_centers(self, result_path):
        
        '''
        plot the representative days in one individual plot.

        Arguments:
            
            result_path: the path of json file that has clustering results

        Returns:

            None
        '''        
        time_length = range(24)

        # set the font
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }    

        # get cluster centers
        centers_dict = self.get_cluster_centers(result_path)

        if self.case_type == 'NE' or self.case_type == 'FE':

            f,ax = plt.subplots(figsize = ((16,6)))
            for key, value in centers_dict.items():
                ax.plot(time_length, value, label = f'cluster_{key}')

            ax.set_xlabel('Time(h)',font = font1)
            ax.set_ylabel('Capacity factor',font = font1)
            
            # save the figures
            folder_path = f'{self.case_type}_case_study/clustering_figures'
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            figname = str(pathlib.Path.cwd().joinpath(f'{self.simulation_data.case_type}_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_whole_centers.jpg'))
            plt.savefig(figname, dpi = 300)

        else:
            # this is for plotting RE centers in one plot. 
            f,(ax1,ax2) = plt.subplots(2,1)
            for key, value in centers_dict.items():
                ax1.plot(time_length, value[0])
                ax2.plot(time_length, value[1])

            ax1.set_ylabel('Capacity factor',font = font1)
            ax2.set_ylabel('Capacity factor',font = font1)
            ax1.set_xlabel('Time(h)',font = font1)
            ax2.set_xlabel('Time(h)',font = font1)
            ax1.set_title('Dispatch')
            ax2.set_title('Wind')
            
            folder_path = f'{self.case_type}_case_study/clustering_figures'
            if not os.path.isdir(folder_path):
                os.mkdir(folder_path)
            figname = str(pathlib.Path.cwd().joinpath(folder_path,f'{self.simulation_data.case_type}_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_whole_centers.jpg'))
            plt.savefig(figname, dpi = 300)

        return