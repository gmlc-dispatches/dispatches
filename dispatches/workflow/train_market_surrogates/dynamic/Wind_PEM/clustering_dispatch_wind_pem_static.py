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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16,12)
import os
import re
import pickle
import json
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

'''
This code do static clustering (P_grid, P_pem, P_wind)

No time series clustering here. We will use clustering (k-means) from scikit learn.
'''


class ClusteringDispatchWind:
    def __init__(self, dispatch_data, wind_data, input_data, wind_gen, num_clusters):
        '''
        Initializes the bidder object.

        Arguments:
            dispatch_data: csv files with the dispatch power data

            wind_data: csv files with wind profile

            wind_gen: name of wind generator

        Return:
            None
        '''
        self.dispatch_data = dispatch_data
        self.wind_data = wind_data
        self.input_data = input_data
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


    def read_dispatch_data(self):

        '''
        read clustering data from dispatch csv files
        
        Aruguments:
            None

        Return: 
            numpy array with dispatch data.
        '''

        # One sim year data is one row, read the target rows.
        df_dispatch = pd.read_csv(self.dispatch_data)

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
        self.years = len(dispatch_array)

        return dispatch_array


    def read_wind_data(self):

        '''
        The length of the dispatch profile and wind profile are in the same length
        which are 366*24 = 8784 days.

        read the wind data without scaling.
        '''
        # read the csv wind file        
        total_wind_profile = pd.read_csv(self.wind_data)

        # select the wind generator in the dataframe
        selected_wind_data = total_wind_profile[self.wind_gen].to_numpy()

        # # scale the array to [0,1]
        # scaled_selected_wind_data = selected_wind_data/self.wind_gen_pmax

        return selected_wind_data


    def calculate_PEM_cf(self):
        '''
        calculate the pem cf for each simulation.

        return the array with sacling.
        '''
        # read dispatch data
        dispatch_array = self.read_dispatch_data()

        # read the input data
        df_input_data = pd.read_hdf(self.input_data)
        # return the number of columns in the df, that is the dimension of the input space. 
        num_col = df_input_data.shape[1]
        num_row = df_input_data.shape[0]

        # drop the first column, which is the indexes
        input_array = df_input_data.iloc[list(range(num_row)), list(range(1,num_col))].to_numpy()
        
        # read the wind profile
        wind_data = self.read_wind_data()

        real_pem_elec_cf = []
        # calcluate the pem power for each simulation
        for idx in range(len(dispatch_array)):
            pem_elec = wind_data - dispatch_array[idx]
            # sometimes we have small pem and pem_elec > pem_pmax
            # input_array[idx][1] is the pem max power for simulation idx.
            real_pem_elec = np.clip(pem_elec, 0, input_array[idx][1])
            real_pem_elec_cf.append(real_pem_elec/input_array[idx][1])     # scale by Pmax_Pem
            # real_pem_elec_cf.append(real_pem_elec/self.wind_gen_pmax)      # scale by Pmax wind
        
        return real_pem_elec_cf


    def transform_data(self):

        '''
        shape the data to the format that scikit learn can read. No filter again in this case. 

        (dispatch_day_data, pem_electricity), pem_electricity = wind_day_data - dispatch_day_data

        Arguments:

        Return:
            train_data: np.arrya for the tslearn package. Dimension = (self.years*364, 24, 1)
            data of full/zero days: [zero_day,full_day]
        '''
        # these two items are not scaled
        dispatch_array = self.read_dispatch_data()
        wind_data = self.read_wind_data()
        
        # this is scaled
        pem_cf = self.calculate_PEM_cf()

        datasets = []

        for year, pem_cf in zip(dispatch_array, pem_cf):
            # scale by the p_max
            pmax_of_year = self.wind_gen_pmax
            scaled_year = year/pmax_of_year
            wind_cf = wind_data/self.wind_gen_pmax

            for h in range(len(scaled_year)):
                # the order is [dispatch_cf, pem_cf, wind_cf]
                datasets.append(np.array([scaled_year[h], pem_cf[h], wind_cf[h]]))
        # use tslearn package to form the correct data structure.
        train_data = np.array(datasets)
        # print(np.shape(datasets))
        # print(train_data[0:])

        return train_data
        

    def cluster_data(self, fname, save_index = True):

        '''
        cluster the data. Save the model to a json file. 
        
        Arguments:

            fname: to save file names
            
            save_index: bool, when True, save the cluster center results in json file.
        
        return:
            label of the data
        '''
        train_data = self.transform_data()
        km = KMeans(n_clusters = self.num_clusters, random_state = 42, n_init = "auto", verbose = 0)
        labels = km.fit_predict(train_data)

        if save_index == True:
            with open (fname, 'wb') as f:
                pickle.dump(km, f)

        return km, labels


    def get_cluster_centers(self, result_path):

        '''
        Get the cluster centers.

        Arguments:

            result_path: the path of clustering model

        Returns:

            centers_list: {cluster_center:[results]} 
        '''

        with open(result_path, 'rb') as f:
            model = pickle.load(f)
        
        centers = model.cluster_centers_

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

        with open(result_path, 'rb') as f:
            model = pickle.load(f)
        
        labels = model.labels_

        label_data_dict = {}
        for idx,lb in enumerate(labels):
            if lb not in label_data_dict:
                label_data_dict[lb] = []
                label_data_dict[lb].append(train_data[idx])
            else:
                label_data_dict[lb].append(train_data[idx])

        return label_data_dict


    def plot_results(self, result_path, train_data):
        
        '''
        Plot box plot for each cluster.

        Arguments: 

            result_path: the path of json file that has clustering results

        Returns:

            None
        '''

        label_data_dict = self._summarize_results(result_path, train_data)
        centers_dict = self.get_cluster_centers(result_path)

        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18,
        }

        outlier_count = {}
        result_dict = {}
        lh = {}
        # result_dict[0] = []     
        # result_dict[1] = []
        # result_dict[2] = []

        for k in range(3):
            result_dict[k] = {}     # 0: grid, 1: pem, 2: wind
            for i in range(20):
                res_array = np.array(label_data_dict[i])
                result_dict[k][i] = []
                for j in res_array:
                    result_dict[k][i].append(j[k])


        # for j in range(20):
        #     print('dispatch', np.mean(result_dict[0][j]), np.max(result_dict[0][j]),np.min(result_dict[0][j]),centers_dict[j][0])
        # 5 clusters in 1 plot
        for s in result_dict:
            result = result_dict[s]
            outlier_count[s] = {}
            lh[s] = {}
            for c in range(4):
                fig_res_list = []
                fig_label = []
                fig_center_list = []
                for m in range(c*5, (c+1)*5):
                    fig_res_list.append(np.array(result[m]).flatten())
                    fig_center_list.append(np.array(centers_dict[m][s]))
                    # calculate the percentage of points in the cluster
                    percentage = np.round(len(result[m])/224/24/366*100, 2)
                    # count the outliers
                    Q1 = np.quantile(np.array(result[m]).flatten(), 0.25)
                    Q3 = np.quantile(np.array(result[m]).flatten(), 0.75)
                    gap = 1.5*(Q3-Q1)
                    lower = np.sum(np.array(result[m]).flatten()<= Q1-gap-0.0001)
                    higher = np.sum(np.array(result[m]).flatten()>= Q3+gap+0.0001)
                    lh[s][m] = (lower, higher)
                    outlier_count[s][m] = np.round((lower+higher)/len(np.array(result[m]))*100,4)
                    fig_label.append(f'cluster_{m}'+'\n'+str(percentage)+'%'+'\n'+str(outlier_count[s][m])+'%')

                f,ax = plt.subplots(figsize = (8,6))
                ax.boxplot(fig_center_list, labels = fig_label, medianprops = {'color':'r'})
                ax.boxplot(fig_res_list, labels = fig_label, medianprops = {'color':'g'})
                ax.set_ylabel('capacity_factor', font = font1)
                custom_lines = [Line2D([0], [0], color='g', lw=2),
                                Line2D([0], [0], color='r', lw=2)]
                ax.legend(custom_lines, ['median', 'mean'])
                if s == 0:
                    figname = f'static clustering grid dispatch box_plot_{c}.jpg'
                    ax.set_title('Dispatch Capacity Factor Boxplot')
                elif s == 1:
                    figname = f'static clustering PEM electricity box_plot_{c}.jpg'
                    ax.set_title('PEM Capacity Factor Boxplot')
                else:
                    figname = f'static clustering wind power box_plot_{c}.jpg'
                    ax.set_title('Wind Capacity Factor Boxplot')
                # plt.savefig will not overwrite the existing file
                plt.savefig(figname,dpi =300)
        
        return


def main():

    num_clusters = 20

    dispatch_data = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_Dispatch_whole.csv'
    wind_data = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Real_Time_wind_hourly.csv'
    input_data = '../../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    wind_gen = '303_WIND_1'
    tsa_task = ClusteringDispatchWind(dispatch_data, wind_data, input_data, wind_gen, num_clusters)
    # dispatch_array = tsa_task.read_dispatch_data()
    # wind_data = tsa_task.read_wind_data()
    # print(wind_data[0:15])
    train_data= tsa_task.transform_data()
    fname = 'static_clustering.pkl'
    # km, labels = tsa_task.cluster_data(fname, save_index = True)
    # with open (fname, 'rb') as f:
    #     model = pickle.load(f)
    # label_dict = tsa_task._summarize_results(fname, train_data)
    # print(model.cluster_centers_)
    # print(model.n_iter_)
    # for i in label_dict:
    #     print(len(label_dict[i]))
    tsa_task.plot_results(fname, train_data)

    # print(np.shape(train_data))
    # tsa_task.find_dispatch_max_min(fname,train_data)
    # tsa_task.wind_dispatch_check(fname)
    # tsa_task.cluster_analysis(fname, train_data)


if __name__ == '__main__':
    main()
