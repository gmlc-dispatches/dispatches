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
write the doc string before starting to code.

This code only do clustering on dispacth power.

Set two clusters: capacity factor = 0 or 1. Filter out 0/1 capacity days before do clustering.
'''

class TSA64K:
    def __init__(self, dispatch_data, metric, years, num_clusters, filter_opt):
        '''
        Initializes the bidder object.

        Arguments:
            dispatch_data: csv files with the dispatch power data

            metric: distance metric (“euclidean” or “dtw”).

            years: The size for the clustering dataset.

            filter: If we need to filter 0/1 capacity days

        Return:
            None
        '''
        self.dispatch_data = dispatch_data
        self.metric = metric 
        self.years = int(years)
        self.num_clusters = num_clusters
        self.filter = filter_opt
        # to do: add typer check for init variables



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

    def _read_input_pmax(self):

        '''
        read the input p_max for each simulation year

        Arguments:
            None

        Return:
            None
        '''
        this_file_path = os.getcwd()
        input_data = os.path.join(this_file_path, '..\\datasets\\prescient_generator_inputs.h5')
        df_input_data = pd.read_hdf(input_data)

        # first column is the p_max, from run_0 to run_64799
        df_pmax = df_input_data.iloc[:,1]
        pmax = df_pmax.to_numpy(dtype = float)
        self.pmax = pmax
        
        return pmax


    def transform_data(self, dispatch_array):

        '''
        shape the data to the format that tslearn can read.

        Arguments:
            dispatch data in the shape of numpy array. (Can be obtained from self.read_data())

        Return:
            train_data: np.arrya for the tslearn package. Dimension = (self.years*364, 24, 1)
            number of full/zero days: np.array([full_day,zero_day])
        '''
    
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

        pmax = self._read_input_pmax()

        datasets = []

        if self.filter_opt == True:
            full_day = 0
            zero_day = 0
            for year,idx in zip(dispatch_years, dispatch_years_index):
                # scale by the p_max
                pmax_of_year = pmax[idx]
                scaled_year = year/pmax_of_year

                # slice the year data into day data(24 hours a day)
                # filter out full/zero capacity days
                for i in range(day_num):
                    day_data = scaled_year[i*time_len:(i+1)*time_len]
                    # count the day of full/zero capacity factor.
                    if sum(day_data) == 0:
                        zero_day += 1
                    # here should be 24 instead of 1.
                    elif sum(day_data) == 24: 
                        full_day += 1
                    else:
                        # datasets = [day_1, day_2,...,day_xxx], day_xxx is np.array([hour0,hour1,...,hour23])
                        datasets.append(day_data)
            
            # use tslearn package to form the correct data structure.
            train_data = to_time_series_dataset(datasets)

        else:
            for year,idx in zip(dispatch_years, dispatch_years_index):
            # scale by the p_max
            pmax_of_year = pmax[idx]
            scaled_year = year/pmax_of_year

            # slice the year data into day data(24 hours a day)
            
            for i in range(day_num):
                day_data = scaled_year[i*time_len:(i+1)*time_len]
                datasets.append(day_data)
        
            # use tslearn package to form the correct data structure.
            train_data = to_time_series_dataset(datasets)
            full_day = 'filter = False'
            zero_day = 'filter = False'

        return train_data, np.array([full_day,zero_day])

    def transform_origin_data(self, dispatch_array):

        '''
        shape the data to the fromat that tslearn can read without filter out 0/1 days.

        Aruguments:
            dispatch data in the shape of numpy array. (Can be obtained from self.read_data())

        Return:
            Readable datasets for the tslearn package.
        '''
        
        datasets = []
        time_len = 24
        # how many years are there in a day. Use the first year's data to calculate. 
        # should have 364 day in a year in our simulation
        day_num = int(np.round(len(dispatch_array[0])/time_len))
        self.day_num = day_num

        # Test on targeted # of years
        dispatch_years = dispatch_array[0:self.years]
        # Need to have the index to do scaling by pmax. 
        dispatch_years_index = self.index[0:self.years]

        for year,idx in zip(dispatch_years, dispatch_years_index):
            # scale by the p_max
            pmax_of_year = self.pmax[idx]
            scaled_year = year/pmax_of_year

            # slice the year data into day data(24 hours a day)
            
            for i in range(day_num):
                day_data = scaled_year[i*time_len:(i+1)*time_len]
                datasets.append(day_data)
        
        # use tslearn package to form the correct data structure.
        train_data = to_time_series_dataset(datasets)

        return train_data, dispatch_years_index

    def cluster_data(self, train_data, data_num, save_index = False):

        '''
        cluster the data. Save the model to a json file. 
        
        Arguments:
            train_data: from self.transform_data
            clusters: number of clusters specified
            data_num: index for saving file name.
            save_index: bool, when True, save the cluster center results in json file.
        
        return:
            result path (json) file. 
        '''

        km = TimeSeriesKMeans(n_clusters = self.num_clusters, metric = self.metric, random_state = 0)
        labels = km.fit_predict(train_data)

        if save_index == True:
            path0 = os.getcwd()
            result_path = os.path.join(path0, f'..\\clustering_results\\result_{self.years}years_{data_num}_{self.num_clusters}clusters_OD.json')
            km.to_json(result_path)
            # print(result_path)

        return result_path


def main():
    
    metric = 'euclidean'
    # In conceptual design problem, the results are clustered from Dispatch_shuffled_data_0.csv, 6400 years, 30 clusters.
    years = 10
    num_clusters = 10
    filter_opt = True

    # we have 64000 simulations shuffled in 10 csv files. 
    # Current results are built in Dispatch_data_shuffled_0.csv

    for i in range(1):
        this_file_path = os.getcwd()
        dispatch_data = os.path.join(this_file_path, f'..\\datasets\\Dispatch_shuffled_data_{i}.csv')
        tsa_task = TSA64K(dispatch_data, metric, years, num_clusters, filter_opt)
        dispatch_array = tsa_task.read_data()
        train_data,day_01 = tsa_task.transform_data(dispatch_array)
        result_path = tsa_task.cluster_data(train_data, i, save_index = False)
        print(day_01)

if __name__ == '__main__':
    main()
