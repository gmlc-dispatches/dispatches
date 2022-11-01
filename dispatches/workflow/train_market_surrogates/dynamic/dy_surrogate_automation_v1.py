import os

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
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
    def __init__(self, dispatch_data_file, input_data_file, num_sims, case_type, fixed_pmax = True):
        '''
        Initialization for the class
        
        Arguments:
		    
		    dispatch_data_file: data path that has the dispatch data
            
            input_data_file: data path that has the input data for sweep

            num_sims: int, number of simulations that we are going to read.

            case_type: str, must be one of 'RE, NE, FE'

            fixed_pmax: bool, default True. If the pmax of the generator is fixed. 
        
        Returns:

            None
        '''
        
        # self.revenue_data_file = revenue_data_file
        self.dispatch_data_file = dispatch_data_file
        self.input_data_file = input_data_file
        self.num_sims = num_sims
        self.case_type = case_type
        self.fixed_pmax = fixed_pmax
        # self.check_case_type()
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


    @property
    def case_type(self):

        '''
        Porperty getter of case_type

        Returns:

            str: the case study type.
        '''

        return self._case_type


    @case_type.setter
    def case_type(self, value):

        '''
        Property setter of case_type

        Arguments:

            value: intended value for case_type

        Returns:
            
            None
        '''

        if not isinstance(value, str):
            raise TypeError(
                f"The value of case_type must be str, but {type(value)} is given."
            )

        if value not in ['RE','NE','FE']:
            raise ValueError(
                f"The case_type must be one of 'RE','NE' or 'FE', but {value} is given."
            )

        self._case_type = value


    @property
    def fixed_pmax(self):

        '''
        Porperty getter of fixed_pmax

        Returns:

            bool: the fixed_pmax bool
        '''

        return self._fixed_pmax


    @fixed_pmax.setter
    def fixed_pmax(self, value):

        '''
        Property setter of fixed_pmax

        Arguments:

            value: intended value for fixed_pmax

        Returns:
            
            None
        '''

        if not isinstance(value, bool):
            raise TypeError(
                f"The fixed_pmax must be bool, but {type(value)} is given."
            )

        self._fixed_pmax = value


    def _read_data_to_array(self):

        '''
        Read the dispatch data from the csv file

        Arguments:
			
            dispatch_data_fil: the file stores dispatch profiles by simulation years

            input_data_file: the file stores input data for parameter sweep

        Returns:
			
            list: [rt_dispatch_array, rt_lmp_array, da_dispatch_array, da_lmp_array]
        '''

        # read the data from excel by sheet names
        df_rt_dispatch = pd.read_excel(self.dispatch_data_file, sheet_name = 'rt_dispatch')
        df_rt_lmp = pd.read_excel(self.dispatch_data_file, sheet_name = 'rt_lmp')
        df_da_dispatch = pd.read_excel(self.dispatch_data_file, sheet_name = 'da_dispatch')
        df_da_lmp = pd.read_excel(self.dispatch_data_file, sheet_name = 'da_lmp')

        # drop the first column
        df_rt_dispatch_data = df_rt_dispatch.iloc[:, 1:]
        df_rt_lmp_data = df_rt_lmp.iloc[:, 1:]
        df_da_dispatch_data = df_da_dispatch.iloc[:, 1:]
        df_da_lmp_data = df_da_lmp.iloc[:, 1:]

        # the first column is the run_index. Put them in an array
        # indexes are the same for all sheets.
        df_index = df_rt_dispatch.iloc[:,0]
        run_index = df_index.to_numpy(dtype = str)

        # save the index in an list.
        # transfer from str to int and put them in a list
        index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            index.append(int(index_num))

        # transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
        rt_dispatch_array = df_rt_dispatch_data.to_numpy(dtype = float)
        rt_lmp_array = df_rt_lmp_data.to_numpy(dtype = float)
        da_dispatch_array = df_da_dispatch_data.to_numpy(dtype = float)
        da_lmp_array = df_da_lmp_data.to_numpy(dtype = float)

        return [rt_dispatch_array, rt_lmp_array, da_dispatch_array, da_lmp_array], index


    # def _remove_bad_data(self):
    #     '''
    #     remove the bad data

    #     This is only temporarily for RE wind_h2 case
    #     '''
    #     data_list, index = self._read_data_to_array()

    #     if self.case_type == 'RE':
    #         bad_idx = [43,54,84,103,155,160,219]
    #         for i in bad_idx:
    #             index.remove(i)
    #         new_data_list = []
    #         for j in data_list:
    #             j_ = np.delete(j, bad_idx, axis = 0)
    #             new_data_list.append(j_)

    #         return new_data_list, index

    #     else:
    #         return data_list, index




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

        data_list, index = self._read_data_to_array()

        # put all the data in a dict
        rt_dispatch_dict = {}
        rt_lmp_dict = {}
        da_dispatch_dict = {}
        da_lmp_dict = {}

        for num, idx in enumerate(index):
            rt_dispatch_dict[idx] = data_list[0][num]
            rt_lmp_dict[idx] = data_list[1][num]
            da_dispatch_dict[idx] = data_list[2][num]
            da_lmp_dict[idx] = data_list[3][num]

        sim_index = list(rt_dispatch_dict.keys())

        # read the input data
        df_input_data = pd.read_hdf(self.input_data_file)
        # return the number of columns in the df
        num_col = df_input_data.shape[1]

        X = df_input_data.iloc[sim_index,list(range(1,num_col))].to_numpy()

        input_data_dict = {}

        for num, x in zip(sim_index, X):
            input_data_dict[num] = x

        self._dispatch_dict = rt_dispatch_dict
        self._input_data_dict = input_data_dict

        # put all the data in one dict
        data_dict = {}
        data_dict['rt_dispatch'] = rt_dispatch_dict
        data_dict['rt_lmp'] = rt_lmp_dict
        data_dict['da_dispatch'] = da_dispatch_dict
        data_dict['da_lmp'] = da_lmp_dict

        return data_dict, input_data_dict
	

    def _calculate_revenue(self):
        
        '''
        Calculate the revenue from the sweep data

        Arguments:

            None

        Return:

            rev_dict: dictionary that has the revenue data, {run_index: rev(int)}
        '''


        # the rt and da dispatch and lmp data are in data_list returned by self.read_data_to_dict
        
        data_dict, input_data_dict = self.read_data_to_dict()
        da_dispatch_dict = data_dict['da_dispatch']
        rt_dispatch_dict = data_dict['rt_dispatch']
        da_lmp_dict = data_dict['da_lmp']
        rt_lmp_dict= data_dict['rt_lmp']

        index_list = list(self._dispatch_dict.keys())

        revenue_dict = {}
        for idx in index_list:
            da_dispatch_data_array = da_dispatch_dict[idx]
            da_lmp_data_array = da_lmp_dict[idx]
            rt_dispatch_data_array = rt_dispatch_dict[idx]
            rt_lmp_data_array = rt_lmp_dict[idx]

            revenue = 0
            for rt_dispatch, rt_lmp, da_dispatch, da_lmp in zip(da_dispatch_data_array, da_lmp_data_array, rt_dispatch_data_array, rt_lmp_data_array):
                # the revenue is equal to rt_lmp*(rt_dispatch - da_dispatch) + da_lmp*da_dispatch
                revenue += (rt_dispatch - da_dispatch)*rt_lmp + da_dispatch*da_lmp

            revenue_dict[idx] = revenue

        return revenue_dict


    def _read_pmin(self):

        '''
        Read pmin from input_dict, this function is only for nuclear case study

        Arguments:
    
            dispatch_dict: dictionary stores dispatch data.

            input_dict: dictionary stores input data for parameter sweep

        Returns:
            pmin_dict: {run_index: pmin}
        '''
        if self.fixed_pmax == True:
            self.pmax = 400
            index_list = list(self._dispatch_dict.keys())

            pmin_dict = {}

            for idx in index_list:
                pmin_scaler = self._input_data_dict[idx][1]
                pmin_dict[idx] = self.pmax - self.pmax*pmin_scaler

        else:
            raise ValueError('For NE case study pmax must be fixed.')

        return pmin_dict


    def _read_pmax(self):

        '''
        Read pmax from input_dict according to the case study

        This is for PV + storage case.

        Arguments:
	
            None

        Returns:

            pmax_dict: {run_index: pmax}
        '''

        # if we sweep the pmax as input
        if self.fixed_pmax == False:
            index_list = list(self._dispatch_dict.keys())

            # put the pmax in dictionary.
            pmax_dict = {}

            for idx in index_list:
                pmax = self._pmax
                pmax_dict[idx] = pmax

            return pmax_dict

        else:
            if self.case_type == 'RE':
                # need to implement the generalized function to read pmax accroding to different wind generators.
                pmax = 847 # MW
            return pmax

            # elif self.case_type == 'FE':
            #     # to be decided by discussion


    def _scale_data(self):

        '''
        scale the data by pmax to get capacity factors

        Arguments:

            None

        Returns:

            scaled_dispatch_dict: {run_index: [scaled dispatch data]}
        '''

        if self.case_type == 'RE':
            if self.fixed_pmax == False:
                index_list = list(self._dispatch_dict.keys())

                pmax_dict = self._read_pmax()

                scaled_dispatch_dict = {}

                for idx in index_list:
                    dispatch_year_data = self._dispatch_dict[idx]
                    pmax_year = pmax_dict[idx]

                    # scale the data between [0,1]
                    scaled_dispatch_year_data = dispatch_year_data/pmax_year
                    scaled_dispatch_dict[idx] = scaled_dispatch_year_data
            else:
                pmax = self._read_pmax()
                scaled_dispatch_dict = {}
                for key in self._dispatch_dict:
                    dispatch_year_data = self._dispatch_dict[key]
                    # scale the data between [0,1]
                    scaled_dispatch_year_data = dispatch_year_data/pmax
                    scaled_dispatch_dict[key] = scaled_dispatch_year_data



        elif self.case_type == 'NE':
            
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

        # elif self.case_type == 'FE':
        #     # to be decided by discussion

        
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

        clustering_model = TimeSeriesKMeans(n_clusters = self.num_clusters, metric = self.metric, random_state = 0)
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
        figname = f'{self.case_type}_case_study/clustering_figures/{self.case_type}_result_{self.num_clusters}clusters_{self.simulation_data.num_sims}years_cluster{idx}.jpg'
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
    
    def __init__(self, simulation_data, clustering_model_path, model_type, filter_opt = True):

        '''
        Initialization for the class

        Arguments:
            simulation data: object, composition from ReadData class

            clustering_model_path: path of the saved clustering model

            model_type: str, one of 'frequency' or 'revenue'

            filter_opt: bool, if we are going to filter out 0/1 capacity days

        Return

            None
    	'''
        self.simulation_data = simulation_data
        self.clustering_model_path = clustering_model_path
        self.model_type = model_type
        self.filter_opt = filter_opt
        self._time_length = 24
        
        if self.model_type == 'frequency':
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
    def model_type(self):
        '''
        Porperty getter of model_type

        Arguments:
            None

        Returns:
            model_type
        '''
        
        return self._model_type


    @model_type.setter
    def model_type(self, value):
        '''
        Porperty setter of model_type
        
        Arguments:
            value: str, one of 'frequency' or 'revenue'

        Returns:
            None
        '''
        
        if not isinstance(value, str):
            raise TypeError(
                f"The clustering_model_path must be str, but {type(value)} is given."
            )

        if value not in ['frequency', 'revenue']:
            raise ValueError(
                f"The model_type must be one of 'freqency' of 'revenue'."
            )
        self._model_type = value


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

            return(dispatch_frequency_dict)
        
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

                dispatch_frequency_dict[idx] = []

                for key, value in count_dict.items():
                    dispatch_frequency_dict[idx].append(value)

            return dispatch_frequency_dict


    def _transform_dict_to_array_frequency(self):

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


    def _transform_dict_to_array_revenue(self):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (revenue)
        '''
        revenue_dict = self.simulation_data._calculate_revenue()
        index_list = list(revenue_dict.keys())
        x = []
        y = []
        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(revenue_dict[idx])

        return np.array(x), np.array(y)


    def train_NN(self, NN_size):

        '''
        train the NN model

        Arguments: 

            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''

        if self.model_type == 'freqency':
            model = self.train_NN_frequency(NN_size)
            return model

        elif self.model_type == 'revenue':
            model = self.train_NN_revenue(NN_size)
            return model

    def train_NN_frequency(self, NN_size):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''
        x, ws = self._transform_dict_to_array_frequency()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        del NN_size[0]
        del NN_size[-1]

        # train test split
        x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=42)

        # scale the data both x and ws
        xm = np.mean(x_train,axis = 0)
        xstd = np.std(x_train,axis = 0)
        wsm = np.mean(ws_train,axis = 0)
        wsstd = np.std(ws_train,axis = 0)
        x_train_scaled = (x_train - xm) / xstd
        ws_train_scaled = (ws_train - wsm)/ wsstd

        # train a keras MLP (multi-layer perceptron) Regressor model
        model = keras.Sequential(name=self.model_type)
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

        if self.filter_opt == True:
            clusters = self.num_clusters + 2

        else:
            clusters = self.num_clusters

        R2 = []

        for rd in range(0,clusters):
            # compute R2 metric
            wspredict = predict_ws_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
            residual = 1 - SS_res/SS_tot
            R2.append(residual)

        print(R2)

        xmin = list(np.min(x_train_scaled, axis=0))
        xmax = list(np.max(x_train_scaled, axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
            "ws_mean":list(wsm),"ws_std":list(wsstd)}

        self._model_params = data

        return model


    def train_NN_revenue(self, NN_size):

        '''
        train the revenue NN surrogate model.
        print the R2 results.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes)

        Return:

            model: the NN model
        '''

        x, y = self._transform_dict_to_array_revenue()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        del NN_size[0]
        del NN_size[-1]

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # scale the data both x and ws
        xm = np.mean(x_train,axis = 0)
        xstd = np.std(x_train,axis = 0)
        ym = np.mean(y_train,axis = 0)
        ystd = np.std(y_train,axis = 0)
        x_train_scaled = (x_train - xm) / xstd
        y_train_scaled = (y_train - ym)/ ystd

        # train a keras MLP (multi-layer perceptron) Regressor model
        model = keras.Sequential(name=self.model_type)
        model.add(layers.Input(input_layer_size))
        for layer_size in NN_size:
            model.add(layers.Dense(layer_size, activation='sigmoid'))
        model.add(layers.Dense(output_layer_size))
        model.compile(optimizer=Adam(), loss='mse')
        history = model.fit(x=x_train_scaled, y=y_train_scaled, verbose=0, epochs=500)

        print("Making NN Predictions...") 

        # normalize the data
        x_test_scaled = (x_test - xm) / xstd
        y_test_scaled = (y_test - ym) / ystd

        print("Evaluate on test data")
        evaluate_res = model.evaluate(x_test_scaled, y_test_scaled)
        print(evaluate_res)
        predict_y = np.array(model.predict(x_test_scaled))
        predict_y_unscaled = predict_y*ystd + ym

        # calculate R2
        ypredict = predict_y_unscaled.transpose()
        SS_tot = np.sum(np.square(y_test.transpose() - ym))
        SS_res = np.sum(np.square(y_test.transpose() - ypredict))
        R2 = 1 - SS_res/SS_tot

        print('The R2 in validation is ', R2)

        xmin = list(np.min(x_train_scaled,axis=0))
        xmax = list(np.max(x_train_scaled,axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax, "y_mean":ym,"y_std":ystd}

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

        if self.model_type == 'frequency':
            NN_default_model_path = f'NN_model_params_keras_scaled/keras_{self.simulation_data.case_type}_dispatch_frequency_sigmoid'
            NN_default_param_path = f'NN_model_params_keras_scaled/keras_{self.simulation_data.case_type}_dispatch_frequency_params.json'
        else:
            NN_default_model_path = f'NN_model_params_keras_scaled/keras_{self.simulation_data.case_type}_revenue_sigmoid'
            NN_default_param_path = f'NN_model_params_keras_scaled/keras_{self.simulation_data.case_type}_revenue_params.json'

        # NN_model_path == none
        if NN_model_path == None:
            # save the NN model
            model_save_path = os.path.join(this_file_path, NN_default_model_path)
            model.save(model_save_path)

            if NN_param_path == None:
                # save the sacling parameters
                param_save_path = os.path.join(this_file_path, NN_default_param_path)
                with open(param_save_path, 'w') as f:
                    json.dump(self._model_params, f)
            else:
                with open(NN_param_path, 'w') as f:
                    json.dump(self._model_params, f)

        else:
            model.save(NN_model_path)
            if NN_param_path == None:
                param_save_path = os.path.join(this_file_path, NN_default_param_path)
                with open(param_save_path, 'w') as f:
                    json.dump(self._model_params, f)
            else:
                with open(NN_param_path, 'w') as f:
                    json.dump(self._model_params, f)


    def plot_R2_results(self, NN_model_path = None, NN_param_path = None, fig_name = None):

        '''
        Visualize the R2 result

        Arguments: 

            train_data: list, [x, ws] where x is the input of NN and ws is output. 

            NN_model_path: the path of saved NN model

            NN_param_path: the path of saved NN params

        '''
        this_file_path = os.getcwd()
        font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
            }
        if self.model_type == 'frequency':
            
            x, ws = self._transform_dict_to_array_frequency()
            # use a different random_state from the training
            x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=0)

            if NN_model_path == None:
                # load the NN model from default path
                model_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_dispatch_frequency_sigmoid')
                NN_model = keras.models.load_model(model_save_path)
            else:
                # load the NN model from the given path
                NN_model = keras.models.load_model(NN_model_path)

            if NN_param_path == None:
                # load the NN parameters from default path
                param_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled/keras_training_parameters_ws_scaled.json')
                with open(param_save_path) as f:
                    NN_param = json.load(f)
            else:
                # load the NN parameters from the given path
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


        if self.model_type == 'revenue':

            x, y = self._transform_dict_to_array_revenue()
            # use a different random_state from the training
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            if NN_model_path == None:
                # load the NN model from default path
                model_save_path = os.path.join(this_file_path, '')
                NN_model = keras.models.load_model(model_save_path)
            else:
                NN_model = keras.models.load_model(NN_model_path)

            if NN_param_path == None:
                # load the NN parameters
                param_save_path = os.path.join(this_file_path, '')
                with open(param_save_path) as f:
                    NN_param = json.load(f)
            else:
                with open(NN_param_path) as f:
                    NN_param = json.load(f)

            # scale data
            xm = NN_param['xm_inputs']
            xstd = NN_param['xstd_inputs']
            ym = NN_param['y_mean']
            ystd = NN_param['y_std']

            x_test_scaled = (x_test - xm)/xstd
            pred_y = NN_model.predict(x_test_scaled)
            pred_y_unscaled = pred_y*ystd + ym

            # compute R2
            ypredict = pred_y_unscaled.transpose()
            SS_tot = np.sum(np.square(y_test.transpose() - ym))
            SS_res = np.sum(np.square(y_test.transpose() - ypredict))
            R2 = 1 - SS_res/SS_tot
            print(R2)

            # plot results.
            fig, axs = plt.subplots()
            fig.text(0.0, 0.5, 'Predicted revenue/$', va='center', rotation='vertical',font = font1)
            fig.text(0.4, 0.05, 'True revenue/$', va='center', rotation='horizontal',font = font1)
            fig.set_size_inches(10,10)

            yt = y_test.transpose()
            yp = pred_y_unscaled.transpose()

            axs.scatter(yt,yp,color = "green",alpha = 0.5)
            axs.plot([min(yt),max(yt)],[min(yt),max(yt)],color = "black")
            axs.set_title(f'Revenue',font = font1)
            axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt),0.75*max(yt)),font = font1)    

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)

            if fig_name == None:
                plt.savefig("NE_case_study\\R2_figures\\automation_plot_test_revenue_{}.png".format(self.num_clusters),dpi =300)
            else:
                fig_name_ = fig_name
                plt.savefig(f"{fig_name_}",dpi =300)




def main():

    current_path = os.getcwd()

    # for RE_H2
    dispatch_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2_new/Dispatch_data_RE_H2_whole.xlsx'
    input_data_path = '../../../../../datasets/results_renewable_sweep_Wind_H2_new/sweep_parameters_results_RE_H2_whole.h5'
    case_type = 'RE'
    num_clusters = 20
    num_sims = 224

    # for NE
    # dispatch_data_path = '../../../../../datasets/results_nuclear_sweep/Dispatch_data_NE_whole.xlsx'
    # input_data_path = '../../../../../datasets/results_nuclear_sweep/sweep_parameters_results_nuclear_whole.h5'
    # case_type = 'NE'
    # num_clusters = 30
    # num_sims = 192

    # whole datasets
    dispatch_data =  dispatch_data_path
    input_data = input_data_path

    # test TimeSeriesClustering
    simulation_data = SimulationData(dispatch_data, input_data, num_sims, case_type)

    # # for RE_H2 case study clustering need to be done in 2-d (dispatch + wind), so I do this in another script.
    # clusteringtrainer = TimeSeriesClustering(num_clusters, simulation_data)
    # train_data = clusteringtrainer._transform_data()
    # clustering_model = clusteringtrainer.clustering_data()
    # RE_PV_path = f'RE_PV_case_study/clustering_results/RE_PV_result_{num_sims}years_{num_clusters}clusters_OD.json'
    # result_path = clusteringtrainer.save_clustering_model(clustering_model, fpath = RE_PV_path)
    # for i in range(num_clusters):
    #     clusteringtrainer.plot_results(result_path, i)
    # outlier_count = clusteringtrainer.box_plots(result_path)
    # clusteringtrainer.plot_centers(result_path)


    # TrainNNSurrogates, revenue
    model_type = 'revenue'
    clustering_model_path = 'str'
    NNtrainer = TrainNNSurrogates(simulation_data, clustering_model_path, model_type)
    model = NNtrainer.train_NN([4,100,100,1])
    NN_rev_model_path = os.path.join(current_path, f'automation_{case_type}_revenue')
    NN_rev_param_path = os.path.join(current_path, f'automation_{case_type}_revenue.json')
    NNtrainer.save_model(model, NN_rev_model_path, NN_rev_param_path)
    NNtrainer.plot_R2_results(NN_rev_model_path, NN_rev_param_path, fig_name = f'{case_type}_revenue')

    # # TrainNNSurrogates, dispatch frequency
    # model_type = 'frequency'
    # clustering_model_path = 'RE_H2_30clusters.json'
    # NNtrainer = TrainNNSurrogates(simulation_data, clustering_model_path, model_type, filter_opt = False)
    # model = NNtrainer.train_NN([4,75,75,30])
    # NN_frequency_model_path = os.path.join(current_path, f'Wind_PEM\\automation_{case_type}_dispatch_frequency_{num_clusters}clusters')
    # NN_frequency_param_path = os.path.join(current_path, f'Wind_PEM\\automation_{case_type}__dispatch_frequency_{num_clusters}clusters_params.json')
    # NNtrainer.save_model(model, NN_frequency_model_path, NN_frequency_param_path)
    # NNtrainer.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'{case_type}_frequency.jpg')



if __name__ == "__main__":
    main()