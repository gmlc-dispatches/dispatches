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
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans,silhouette_score
import matplotlib.pyplot as plt
import os
import re

'''
This code do clustering over wind+PV+dispatches

filter out dispatch profile days with capacity factor always = 0/1 days

clustering data (dispatch, wind, pv), 3*24 array

dispatch: (d1, d2,..., d24)

wind: (w1, w2,..., w24)

pv: (pv1, pv2,..., pv24)

'''

class TSA64K:
    def __init__(self, dispatch_data, wind_data, pv_data, metric, years, wind_gen, pv_gen):
        '''
        Initializes the bidder object.

        Arguments:
            dispatch_data: csv files with the dispatch power data

            wind_data: csv files with wind profile

            pv_data: csv files with pv profile

            metric: distance metric (“euclidean” or “dtw”).

            years: The size for the clustering dataset.

            wind_gen: name of wind generator

            pv_gen: name of pv generator

        Return:
            None
        '''
        self.dispatch_data = dispatch_data
        self.wind_data = wind_data
        self.pv_data = pv_data
        self.metric = metric
        self.years = int(years)
        self.wind_gen = wind_gen
        self.pv_gen = pv_gen

        # make a dict key = generator name, value = pmax
        _wind_gen_pmax = {}
        _wind_gen_name = ['309_WIND_1', '317_WIND_1', '303_WIND_1', '122_WIND_1']
        _win_gen_pmax_list = [148.3, 799.1, 847, 713.5]
        _pv_gen_pmax = {}
        _pv_gen_name = ['320_PV_1', '314_PV_1', '314_PV_2', '313_PV_1', '314_PV_3', '314_PV_4', '313_PV_2',\
                       '310_PV_1', '324_PV_1', '312_PV_1', '310_PV_2', '324_PV_2', '324_PV_3', '113_PV_1',\
                       '319_PV_1', '215_PV_1', '102_PV_1', '101_PV_1', '102_PV_2', '104_PV_1', '101_PV_2',\
                       '101_PV_3', '101_PV_4', '103_PV_1', '119_PV_1']
        _pv_gen_pmax_list = [51.6, 51.6, 51.6, 95.1, 92.7, 51.6, 93.3,\
                            51.7, 49.7, 94.1, 51.6, 51.6, 51, 93.6,\
                            188.2, 125.1, 25.6, 25.9, 25.3, 26.8, 26.7,\
                            26.2, 25.8, 61.5, 66.6]

        for name, pmax in zip(_wind_gen_name, _win_gen_pmax_list):
            _wind_gen_pmax[name] = pmax

        for name, pmax in zip(_pv_gen_name, _pv_gen_pmax_list):
            _pv_gen_pmax[name] = pmax

            
        # check the wind generator and pv generator name are correct.
        # Assign pmax accroding to generators.
        if self.wind_gen in _wind_gen_name:
            self.wind_gen_pmax = _wind_gen_pmax[self.wind_gen]
        else:
            raise NameError("wind generator name {} is invaild.".format(self.wind_gen))

        if self.pv_gen in _pv_gen_name:
            self.pv_gen_pmax = _pv_gen_pmax[self.pv_gen]
        else:
            raise NameError("PV generator name {} is invaild.".format(self.wind_gen))       

    def read_data(self):

        '''
        read clustering data from dispatch csv files
        
        Aruguments:
            None

        Return: 
            numpy array with dispatch data.
        '''

        ### to save time, need to modify to code to only read the target lines 
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

        return dispatch_array

    def read_input_pmax(self):

        '''
        read the input p_max for each simulation year

        Arguments:
            None

        Return:
            None
        '''

        input_data = 'prescient_generator_inputs.h5'
        df_input_data = pd.read_hdf(input_data)

        # first column is the p_max, from run_0 to run_64799
        df_pmax = df_input_data.iloc[:,1]
        pmax = df_pmax.to_numpy(dtype = float)
        self.pmax = pmax
        
        return


    def read_wind_pv_data(self):

        '''
        length of wind and pv data is 366*24 = 8784 (366 days including Feb 29)
        length of dispatch data is 364*24 = 8736 (364 days, without Jan 1 and Dec 31)
        exclude first and last 24 time points in wind and pv data.
        '''
        wind_file = self.wind_data
        pv_file = self.pv_data

        total_wind_profile = pd.read_csv(wind_file)
        total_pv_profile = pd.read_csv(pv_file)
        selected_wind_data = total_wind_profile[self.wind_gen].to_numpy()
        selected_pv_data = total_pv_profile[self.pv_gen].to_numpy()

        # exclude Jan 1 and Dec 31
        # scale data by pmax to get capacity factors
        selected_wind_data = selected_wind_data[24:8760]/self.wind_gen_pmax
        selected_pv_data = selected_pv_data[24:8760]/self.pv_gen_pmax
        joint_wind_pv = []

        time_len = 24
        day_num = int(len(selected_wind_data)/time_len)
        for i in range(day_num):
            joint_wind_pv.append([selected_wind_data[i*24:(i+1)*24], selected_pv_data[i*24:(i+1)*24]])
        # print(np.shape(joint_wind_pv))
        return joint_wind_pv


    def transform_data(self, dispatch_array, joint_wind_pv, filters = True):

        '''
        shape the data to the format that tslearn can read.

        Arguments:
            dispatch data in the shape of numpy array. (Can be obtained from self.read_data())

        Return:
            train_data: np.arrya for the tslearn package. Dimension = (self.years*364, 24, 1)
            number of full/zero days: np.array([full_day,zero_day])
        '''
        
        datasets = []
        # number of hours in a representative day
        time_len = 24

        # should have 364 day in a year in our simulation
        day_num = int(np.round(len(dispatch_array[0])/time_len))
        self.day_num = day_num

        # How many years are there in dispatch_array is depending on the self.dispatch_data.
        # We use the target number of years (self.years) to do the clustering
        dispatch_years = dispatch_array[0:self.years]

        # Need to have the index to do scaling by pmax. 
        dispatch_years_index = self.index[0:self.years]

        '''
        pmax is from the input data file.
        pmax is transfered from pd.Dataframe
        0        177.50
        1        266.25
        2        355.00
        .        .
        .        .
        .        .
        64799    443.75
        because the sequence of data in the 'self.dispatch_data' is not from 1,2,3,...10000
        it is 1,10,11,...,100,101,...
        So we record every simulation year's run index, and match them in the pmax. 
        '''
        full_day = 0
        zero_day = 0

        for year,idx in zip(dispatch_years, dispatch_years_index):
            # scale by the p_max
            pmax_of_year = self.pmax[idx]
            scaled_year = year/pmax_of_year

            # slice the year data into day data(24 hours a day)

            if filters == True:
            # filter out full/zero capacity days
                for i in range(day_num):
                    dispatch_day_data = scaled_year[i*time_len:(i+1)*time_len]
                    # count the day of full/zero capacity factor.
                    if sum(dispatch_day_data) == 0:
                        zero_day += 1
                    elif sum(dispatch_day_data) == 24:
                        full_day += 1
                    else:
                        # np.shape(datasets) = (num_days, 3, 24))
                        # (wind(1*24), pv(1*24), dispatch(1*24))
                        # joint_wind_pv[i].append(dispatch_day_data)
                        datasets.append([dispatch_day_data,joint_wind_pv[i][0],joint_wind_pv[i][1]])
            else:
                for i in range(day_num):
                    dispatch_day_data = scaled_year[i*time_len:(i+1)*time_len]
                    datasets.append([dispatch_day_data,joint_wind_pv[i][0],joint_wind_pv[i][1]])
        # use tslearn package to form the correct data structure.
        train_data = to_time_series_dataset(datasets)

        return train_data, np.array([full_day,zero_day])
        


    def cluster_data(self, train_data, clusters, data_num, save_index = False):

        '''
        cluster the data. Save the model to a json file. 
        
        Arguments:
            train_data: from self.transform_data
            clusters: number of clusters specified
            data_num: index for saving file name.
            save_index: bool, when True, save the cluster center results in json file.
        
        return:
            label of the data
        '''

        km = TimeSeriesKMeans(n_clusters = clusters, metric = self.metric, random_state = 0)
        labels = km.fit_predict(train_data)

        if save_index == True:
            path0 = os.getcwd()
            result_path = os.path.join(path0, f'result_{self.years}years_{data_num}_{clusters}clusters_OD.json')
            km.to_json(result_path)
            # print(result_path)

        return labels


def main():
    
    metric = 'euclidean'
    years = 10
    num_clusters = 10
    
    for i in range(1):
        dispatch_data = 'Dispatch_shuffled_data_0.csv'
        wind_data = 'DAY_AHEAD_wind.csv'
        pv_data = 'DAY_AHEAD_pv.csv'
        wind_gen = '317_WIND_1'
        pv_gen = '310_PV_1'
        tsa_task = TSA64K(dispatch_data, wind_data, pv_data, metric, years, wind_gen, pv_gen)
        dispatch_array = tsa_task.read_data()
        tsa_task.read_input_pmax()
        joint_windpv = tsa_task.read_wind_pv_data()
        train_data,day_01 = tsa_task.transform_data(dispatch_array,joint_windpv, filters = False)
        # labels = tsa_task.cluster_data(train_data, num_clusters, i, save_index = False)
        print('full capacity days = {}'.format(day_01[0]))
        print('zero capacity days = {}'.format(day_01[1]))
        print(np.shape(train_data))

if __name__ == '__main__':
    main()