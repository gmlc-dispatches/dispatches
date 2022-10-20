import os

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import silhouette_score
from tslearn.utils import to_time_series_dataset
from Time_series_clustering.only_dispatch.filter_01_6400_years import TSA64K
from Time_series_clustering.train_kerasNN.TSA_NN_surrogate_keras import load_cluster_model, calculate_ws, read_input_x, train_NN_surrogate
# from conceptual_design_dynamic.new_full_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE, record_result
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from pyomo.environ import value, SolverFactory
from idaes.core.util import to_json, from_json
import time
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt


# need to find out a place to store the data instead of just put them in the dispatches repo
# temporarily put them in the repo


class SimulationData:
    def __init__(self, dispatch_data_file, input_data_file, num_sims):
        '''
        Initialization for the class
        
        Arguments:
		    
		    dispatch_data_file: data path that has the dispatch data
            
            input_data_file: data path that has the input data for sweep

            num_sims: number of simulations that we are going to read.
        
        Returns:

            None
        '''
        
        self.dispatch_data_file = dispatch_data_file
        self.input_data_file = input_data_file
        self.num_sims = num_sims
        # pmax = 400 for the nuclear generator in RTS-GMLC
        self.pmax = 400
        self.read_data_to_dict()


    @property
    def num_sims(self):

        '''
        Porperty getter of num_sims

        Returns:

            int: number of years for clustering, positive integer.
        '''

        return self._num_sims


    @num_sims.setter
    def num_sims(self, value):

        '''
        Property setter of num_sims

        Arguments:

            value: intended value for num_sims 

        Returns:
			
            None
        '''

        if not isinstance(value, int):
            raise TypeError(
                f"The number of clustering years must be positive integer, but {type(value)} is given."
            )

        if value < 1:
            raise ValueError(
                f"The number of simulation years must be positive integer, but {value} is given."
            )

        self._num_sims = value


    def _read_data_to_array(self):

        '''
        Read the dispatch data from the csv file

        Arguments:
			
            dispatch_data_fil: the file stores dispatch profiles by simulation years

            input_data_file: the file stores input data for parameter sweep

        Returns:

            dispatch_arrry: numpy.ndarray, dispacth data array 

            index: the run index. 
        '''

        df_dispatch = pd.read_csv(self.dispatch_data_file, nrows = self.num_sims)

        # drop the first column
        df_dispatch_data = df_dispatch.iloc[: , 1:]

        # the first column is the run_index. Put them in an array
        df_index = df_dispatch.iloc[:,0]
        run_index = df_index.to_numpy(dtype = str)

        # save the index in an list.
        index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            index.append(int(index_num))

        # transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
        dispatch_array = df_dispatch_data.to_numpy(dtype = float)

        return dispatch_array, index


    def read_data_to_dict(self):
		
        '''
        Transfer the data into dictionary 
		
        Arguments: 
			
            dispatch_data_file: the file stores dispatch profiles by simulation years

            input_data_file: the file stores input data for parameter sweep

        Returns:
			
            dispatch_dict: {run_index:[dispatch data]}

            input_dict: {run_index:[input data]}
        '''

        dispatch_array, index = self._read_data_to_array()

        dispatch_dict = {}

        # project the 
        for num, idx in enumerate(index):
            dispatch_dict[idx] = dispatch_array[num]

        sim_index = list(dispatch_dict.keys())

        # read the input data
        df_input_data = pd.read_hdf(self.input_data_file)

        X = df_input_data.iloc[sim_index,[1,2,3,4]].to_numpy()

        input_data_dict = {}

        for num, x in zip(sim_index, X):
            input_data_dict[num] = x

        self._dispatch_dict = dispatch_dict
        self._input_data_dict = input_data_dict

        return dispatch_dict, input_data_dict
	

    def _read_pmin(self):

        '''
        Read pmin from input_dict, this function is only for nuclear case study

        Arguments:
	
            dispatch_dict: dictionary stores dispatch data.

            input_dict: dictionary stores input data for parameter sweep

        Returns:
            pmin_dict: {run_index: pmin}
        '''

        index_list = list(self._dispatch_dict.keys())

        pmin_dict = {}

        for idx in index_list:
            pmin_scaler = self._input_data_dict[idx][1]
            pmin_dict[idx] = self.pmax - self.pmax*pmin_scaler

        return pmin_dict


    def _scale_data(self):
	
        '''
        scale the data by pmax to get capacity factors

        Arguments:

            None

        Returns:

            scaled_dispatch_dict: {run_index: [scaled dispatch data]}
        '''

        index_list = list(self._dispatch_dict.keys())

        pmin_dict = self._read_pmin()

        scaled_dispatch_dict = {}

        for idx in index_list:
            dispatch_year_data = self._dispatch_dict[idx]
            pmin_year = pmin_dict[idx]

            # scale the data between [0,1] where 0 is the Pmin (Pmax-Ppem)
            # this is for only nuclear case study.
            scaled_dispatch_year_data = (dispatch_year_data - pmin_year)/(self.pmax - pmin_year)
            scaled_dispatch_dict[idx] = scaled_dispatch_year_data

        return scaled_dispatch_dict


    def summarize_statistics(self):
    	
    	'''
    	summarize the statistics of the dispatch data
    	'''

    	return

    def visualize_data(self):
        
        '''
        visualize the statistics of the data
        '''

        return



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

        scaled_dispatch_dict = self.simulation_data._scale_data()

        index_list = list(scaled_dispatch_dict.keys())

        # in each simulation data, count 0/1 days.
        if self.filter_opt == True:
            full_day = 0
            zero_day = 0
            day_dataset = []
            for idx in index_list:
                sim_year_data = scaled_dispatch_dict[idx]
                day_num = int(len(sim_year_data)/self._time_length)

                for day in range(day_num):
                    sim_day_data = sim_year_data[day*24:(day+1)*24]

                    if sum(sim_day_data) == 0:
                        zero_day += 1
                    elif sum(sim_day_data) == 24:
                        full_day += 1
                    else:
                        day_dataset.append(sim_day_data)

            train_data = to_time_series_dataset(day_dataset)
            # print(np.shape(train_data))

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
        labels = clustering_model.fit_predict(train_data)

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

        if fpath == None:
            current_path = os.getcwd()
            result_path =  os.path.join(current_path, f'Time_series_clustering/clustering_result/auto_result_{self.simulation_data.num_sims}years_shuffled_0_{self.num_clusters}clusters_OD.json')
            clustering_model.to_json(result_path)

        else:
            result_path = fpath
            clustering_model.to_json(result_path)

        return result_path


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
        
        labels = cluster_results['model_params']['labels_']

        label_data_dict = {}
        for idx,lb in enumerate(labels):
            if lb not in label_data_dict:
                label_data_dict[lb] = []
                label_data_dict[lb].append(train_data[idx])
            else:
                label_data_dict[lb].append(train_data[idx])

        return label_data_dict


    def plot_results(self, result_path, idx):
        
        '''
        Plot the result data. 

        Arguments: 

            result_path: the path of json file that has clustering results

            idx: int, the index that of the cluster center

        Returns:

            None
        '''

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
        figname = f'NE_case_study/clustering_figures/NE_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_cluster{idx}.jpg'
        if os.path.isfile(figname):
            os.remove(figname)
        plt.savefig(figname, dpi = 300)

        return


    def plot_centers(self, result_path):
        
        '''
        plot the representative days in one plot

        Arguments:
            
            result_path: the path of json file that has clustering results

        Returns:

            None
        '''

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
        figname = f'NE_case_study/clustering_figures/NE_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_whole_centers.jpg'
        plt.savefig(figname, dpi = 300)


    def box_plots(self, result_path):
        
        '''
        Generate box plots for analyzing the clustering resuls.

        Arguments: 

            result_path: the path of json file that has clustering results

        return:
            
            outlier_count: dict, count the number of outliers in each cluster.

        '''


        # read the cluster centers in numpy.ndarray
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
            figname = f"NE_case_study\\clustering_figures\\box_plot_{self.num_clusters}clusters_{p}.jpg"
            # plt.savefig will not overwrite the existing file
            if os.path.isfile(figname):
                os.remove(figname)
            plt.savefig(figname,dpi =300)
            plt.close()
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
        figname = f"NE_case_study\\clustering_figures\\box_plot_{self.num_clusters}clusters_{p}.jpg"
        if os.path.isfile(figname):
            os.remove(figname)
        plt.savefig(figname,dpi =300)
        plt.savefig(figname,dpi =300)
        plt.close()
        
        return outlier_count





class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency
    '''
    
    def __init__(self, simulation_data, clustering_model_path, filter_opt = True):

        '''
        Initialization for the class

        Arguments:
            simulation data: object, composition from ReadData class

            clustering_model_path: path of the saved clustering model

            filter_opt: bool, if we are going to filter out 0/1 capacity days

        Return

            None
    	'''
        self.simulation_data = simulation_data
        self.clustering_model_path = clustering_model_path
        self.filter_opt = filter_opt
        self._time_length = 24
        
        # read and save the clustering model in self.clustering_model
        self.clustering_model = self._read_clustering_model(self.clustering_model_path)

        # read the number of clusters from the clustering model
        self.num_clusters = self.clustering_model.n_clusters


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
    def clustering_model_path(self):
        '''
        Porperty getter of clustering_model_path

        Arguments:
            None

        Returns:
            clustering_model_path
        '''
        
        return self._clustering_model_path


    @clustering_model_path.setter
    def clustering_model_path(self, value):
        '''
        Porperty setter of clustering_model_path
        
        Arguments:
            value: str, path of the clustering model

        Returns:
            None
        '''
        
        if not isinstance(value, str):
        	raise TypeError(
        		f"The clustering_model_path must be str, but {type(value)} is given."
            )
        self._clustering_model_path = value


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



    def _read_clustering_model(self, clustering_model_path):

        '''
        Read the time series clustering model from the given path

        Arguments:

            path of clustering model

        Returns:

            Clustering model
        '''
        clustering_model = TimeSeriesKMeans.from_json(clustering_model_path)

        return clustering_model


    def _generate_label_data(self):

        '''
        Calculate the labels for NN training. 

        Arguments:

            None

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        scaled_dispatch_dict = self.simulation_data._scale_data()
        sim_index = list(scaled_dispatch_dict.keys())
        single_day_dataset = {}
        dispatch_frequency_dict = {}
		
		# filter out 0/1 days in each simulaion data
        if self.filter_opt == True:
            for idx in sim_index:
                sim_year_data = scaled_dispatch_dict[idx]
                single_day_dataset[idx] = []
                # calculate number of days in a simulation
                day_num = int(len(sim_year_data)/self._time_length)
                day_0 = 0
                day_1 = 0
                for day in range(day_num):
                    sim_day_data = sim_year_data[day*self._time_length:(day+1)*self._time_length]
                    if sim_day_data.sum() == 0:
                        day_0 += 1
                    elif sim_day_data.sum() == 24:
                        day_1 += 1
                    else:
                        single_day_dataset[idx].append(sim_day_data)
			
				# frequency of 0/1 days
                ws0 = day_0/day_num
                ws1 = day_1/day_num


                if len(single_day_dataset[idx]) == 0:
                    labels = np.array([])

                else:
                    to_pred_data = to_time_series_dataset(single_day_dataset[idx])
                    labels = self.clustering_model.predict(to_pred_data)

                elements, count = np.unique(labels,return_counts=True)

                pred_result_dict = dict(zip(elements, count))
                count_dict = {}
                for j in range(self.num_clusters):
                    if j in pred_result_dict.keys():
                        count_dict[j] = pred_result_dict[j]/day_num
                    else:
                        count_dict[j] = 0

                # the first element in w is frequency of 0 cf days
                dispatch_frequency_dict[idx] = [ws0]

                for key, value in count_dict.items():
                    dispatch_frequency_dict[idx].append(value)

                # the last element in w is frequency of 1 cf days
                dispatch_frequency_dict[idx].append(ws1)

            return dispatch_frequency_dict
        
        # filter_opt = False then we do not filter 0/1 days
        else:
            for idx in sim_index:
                sim_year_data = scaled_dispatch_dict[idx]
                single_day_dataset[idx] = []
                # calculate number of days in a simulation
                day_num = int(len(sim_year_data)/self._time_length)
                for day in range(day_num):
                    single_day_dataset[idx].append(sim_day_data)

                to_pred_data = to_time_series_dataset(single_day_dataset[idx])
                labels = self.clustering_model.predict(to_pred_data)

                elements, count = np.unique(labels,return_counts=True)
                pred_result_dict = dict(zip(elements, count))
                count_dict = {}
				
                for j in range(self.num_clusters):
                    if j in pred_result_dict.keys():
                        count_dict[j] = pred_result_dict[j]/day_num
                    else:
                        count_dict[j] = 0

                for key, value in count_dict.items():
                    dispatch_frequency_dict[idx].append(value)

            return dispatch_frequency_dict


    def _transform_dict_to_array(self):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (dispatch frequency)
        '''

        dispatch_frequency_dict = self._generate_label_data()
		
        index_list = list(self.simulation_data._dispatch_dict.keys())

        x = []
        y = []

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(dispatch_frequency_dict[idx])

        return np.array(x), np.array(y)


    def train_NN(self, NN_size):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        return:

            None
        '''
        x, ws = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        del NN_size[0]
        del NN_size[-1]


        x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=42)

        # scale the data both x and ws
        xm = np.mean(x_train,axis = 0)
        xstd = np.std(x_train,axis = 0)
        wsm = np.mean(ws_train,axis = 0)
        wsstd = np.std(ws_train,axis = 0)
        x_train_scaled = (x_train - xm) / xstd
        ws_train_scaled = (ws_train - wsm)/ wsstd

        # train a keras MLP (multi-layer perceptron) Regressor model
        model = keras.Sequential(name='dispatch_frequency')
        model.add(layers.Input(input_layer_size))
        for layer_size in NN_size:
            model.add(layers.Dense(layer_size, activation='sigmoid'))
        model.add(layers.Dense(output_layer_size))
        model.compile(optimizer=Adam(), loss='mse')
        history = model.fit(x=x_train_scaled, y=ws_train_scaled, verbose=0, epochs=500)

        print("Making NN Predictions...") 

        # normalize the data
        x_test_scaled = (x_test - xm) / xstd
        ws_test_scaled = (ws_test - wsm) / wsstd

        print("Evaluate on test data")
        evaluate_res = model.evaluate(x_test_scaled, ws_test_scaled)
        print(evaluate_res)
        predict_ws = np.array(model.predict(x_test_scaled))
        predict_ws_unscaled = predict_ws*wsstd + wsm

        test_R2 = []

        if self.filter_opt == True:
            clusters = self.num_clusters + 2

        else:
            clusters = self.num_clusters

        for rd in range(0,clusters):
            # compute R2 metric
            wspredict = predict_ws_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
            residual = 1 - SS_res/SS_tot
            test_R2.append(residual)

        accuracy_dict = {"R2":test_R2}

        print(test_R2)

        xmin = list(np.min(x_train_scaled, axis=0))
        xmax = list(np.max(x_train_scaled, axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
            "ws_mean":list(wsm),"ws_std":list(wsstd)}

        self._model_params = data

        return model


    def save_model(self, model, NN_model_path = None, NN_param_path = None):

        '''
        Save the model to the path which can be specified by the user. 

        Arguments:

            model: trained model from self.train_NN()

            fpath: if fpath == None, save to default path. 

        Return:

            None
		'''

        this_file_path = os.getcwd()
        if NN_model_path == None:
            # save the NN model
            model_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_dispatch_frequency_sigmoid')
            model.save(model_save_path)

            if NN_param_path == None:
                # save the sacling parameters
                param_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_training_parameters_ws_scaled.json')
                with open(param_save_path, 'w') as f:
                    json.dump(self._model_params, f)
            else:
                with open(NN_param_path, 'w') as f:
                    json.dump(self._model_params, f)

        else:
            model.save(NN_model_path)
            if NN_param_path == None:
                param_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_training_parameters_ws_scaled.json')
                with open(param_save_path, 'w') as f:
                    json.dump(self._model_params, f)
            else:
                with open(NN_param_path, 'w') as f:
                    json.dump(self._model_params, f)


    # In progress 
    def plot_R2_results(self, NN_model_path = None, NN_param_path = None, fig_name = None):

        '''
        Visualize the R2 result

        Arguments: 

            train_data: list, [x, ws] where x is the input of NN and ws is output. 

            NN_model_path: the path of saved NN model

            NN_param_path: the path of saved NN params

        '''
        this_file_path = os.getcwd()

        x, ws = self._transform_dict_to_array()
        x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=42)

        if NN_model_path == None:
            # load the NN model
            model_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_dispatch_frequency_sigmoid')
            NN_model = keras.models.load_model(model_save_path)

        else:
            NN_model = keras.models.load_model(NN_model_path)

        if NN_param_path == None:
            # load the NN parameters
            param_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_training_parameters_ws_scaled.json')
            with open(param_save_path) as f:
                NN_param = json.load(f)
        else:
            with open(NN_param_path) as f:
                NN_param = json.load(f)

        # scale data
        xm = NN_param['xm_inputs']
        xstd = NN_param['xstd_inputs']
        wsm = NN_param['ws_mean']
        wsstd = NN_param['ws_std']

        x_test_scaled = (x_test - xm)/xstd
        pred_ws = NN_model.predict(x_test_scaled)
        pred_ws_unscaled = pred_ws*wsstd + wsm

        if self.filter_opt == True:
            num_clusters = self.num_clusters + 2

        else:
            num_clusters = self.num_cluster

        # calculate the R2 for each representative day
        R2 = []
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }
        for rd in range(num_clusters):
            # compute R2 metric
            wspredict = pred_ws_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
            residual = 1 - SS_res/SS_tot
            R2.append(residual)
        print(R2)
        # plot the figure
        for i in range(num_clusters):
            fig, axs = plt.subplots()
            fig.text(0.0, 0.5, 'Predicted dispatch frequency', va='center', rotation='vertical',font = font1)
            fig.text(0.4, 0.05, 'True dispatch frequency', va='center', rotation='horizontal',font = font1)
            fig.set_size_inches(10,10)

            wst = ws_test.transpose()[i]
            wsp = pred_ws_unscaled.transpose()[i]

            axs.scatter(wst,wsp,color = "green",alpha = 0.5)
            axs.plot([min(wst),max(wst)],[min(wst),max(wst)],color = "black")
            axs.set_title(f'cluster_{i}',font = font1)
            axs.annotate("$R^2 = {}$".format(round(R2[i],3)),(min(wst),0.75*max(wst)),font = font1)


            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)
            if fig_name == None:
                plt.savefig("NE_case_study\\R2_figures\\automation_plot_test_cluster{i}.png".format(i),dpi =300)
            else:
                fig_name_ = fig_name + f'_cluster_{i}'
                plt.savefig(f"NE_case_study\\R2_figures\\{fig_name_}",dpi =300)

        return




def main():

    current_path = os.getcwd()

    # whole datasets (192 sims)
    dispatch_data = os.path.join(current_path, 'results_nuclear_sweep/Dispatch_data_NE_whole.csv')
    input_data = os.path.join(current_path, 'results_nuclear_sweep/sweep_parameters_results_nuclear_whole.h5')

    # seperate dataset (48 sims)
    # dispatch_data = os.path.join(current_path, 'results_nuclear_sweep/Dispatch_data_NE_results_nuclear_sweep_10_500.csv')
    # input_data = os.path.join(current_path, 'results_nuclear_sweep/input_data/sweep_parameters_results_nuclear_sweep_10_500.h5')

    num_clusters = 30
    num_sims = 192

    # test TimeSeriesClustering
    simulation_data = SimulationData(dispatch_data, input_data, num_sims)
    # scaled_data = simulation_data._scale_data()


    clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data)
    train_data = clusteringtrainer._transform_data()

    clustering_model = clusteringtrainer.clustering_data()
    NE_path = f'NE_case_study/clustering_results/NE_result_{num_sims}years_{num_clusters}clusters_OD.json'
    result_path = clusteringtrainer.save_clustering_model(clustering_model, fpath = NE_path)

    for i in range(num_clusters):
        clusteringtrainer.plot_results(result_path, i)
    outlier_count = clusteringtrainer.box_plots(result_path)
    clusteringtrainer.plot_centers(result_path)
    # test class TrainNNSurrogates
    NNtrainer = TrainNNSurrogates(simulation_data, NE_path)
    model = NNtrainer.train_NN([4,50,50,num_clusters+2])
    NN_model_path = os.path.join(current_path, f'NE_case_study\\automation_NE_{num_sims}sims_{num_clusters}clusters')
    NN_param_path = os.path.join(current_path, f'NE_case_study\\automation_NE_params_{num_sims}sims_{num_clusters}clusters.json')
    NNtrainer.save_model(model, NN_model_path, NN_param_path)
    NNtrainer.plot_R2_results(NN_model_path, NN_param_path,fig_name = f'NE_{num_sims}sims_{num_clusters}clusters')

    print(outlier_count)



if __name__ == "__main__":
    main()