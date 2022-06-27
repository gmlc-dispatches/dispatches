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


# write the doc string before starting to code.

# This code only do clustering on dispacth power delta.

class TSA64K:
    def __init__(self, dispatch_data, metric, years):
        '''
        Initializes the bidder object.

        Arguments:
            dispatch_data: csv files with the dispatch power data

            metric: distance metric (“euclidean” or “dtw”).

            years: The size for the clustering dataset.

        Return:
            None
        '''
        self.dispatch_data = dispatch_data
        self.metric = metric 
        self.years = int(years)

    def read_data(self):

        '''
        read clustering data from dispatch csv files
        
        Aruguments:
            None

        Return: 
            numpy array with dispatch data.
        '''

        df_dispatch = pd.read_csv(self.dispatch_data)

        # drop the first column
        df_dispatch_data = df_dispatch.iloc[: , 1:]

        # the first column is the run_index. Put them in an array
        df_index = df_dispatch.iloc[:,0]
        run_index = df_index.to_numpy(dtype = str)

        # run is string 'run_xxxx', make xxxx into a int and put them in list
        self.index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            self.index.append(int(index_num))

        # print(self.index)
        # transfer the data to the np.array, dimension of test_years*8736(total dispatch hours in one year)
        dispatch_array = df_dispatch_data.to_numpy(dtype = float)

        return dispatch_array

    def read_input_pmax(self):

        input_data = 'prescient_generator_inputs.h5'
        df_input_data = pd.read_hdf(input_data)
        df_pmax = df_input_data.iloc[:,1]
        pmax = df_pmax.to_numpy(dtype = float)
        self.pmax = pmax
        
        return


    def transform_data(self, dispatch_array):

        '''
        shape the data to the fromat that tslearn can read.

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

    def cluster_data(self, train_data, clusters, data_num, save_index = False):
        '''
        cluster the data. Save the model to a json file. 

        return:
            silhouette score and label
        '''

        km = TimeSeriesKMeans(n_clusters = clusters, metric = self.metric, random_state = 0)
        labels = km.fit_predict(train_data)
        # sc = silhouette_score(train_data, labels, metric = self.metric)
        # print(labels)
        # print(train_data.shape)
        if save_index == True:
            path0 = os.getcwd()
            result_path = os.path.join(path0, f'result_{self.years}years_{data_num}_{clusters}clusters_OD.json')
            km.to_json(result_path)
            # print(result_path)

        return labels

    def plot_origin_data(self, lmp_array, dispatch_array):
        '''
        plot the original data

        return:
            None
        '''

        # Test on 1 year 
        lmp_year_0 = lmp_array[0]
        dispatch_year_0 = dispatch_array[0]
        time_len = 24

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,9))
        for i in range(self.day_num):
            lmp_i = lmp_year_0[i*time_len:(i+1)*time_len]
            dis_i = dispatch_year_0[i*time_len:(i+1)*time_len]
            for j,k in enumerate(lmp_i):
                if k > 100:
                    lmp_i[j] = 100
                else:
                    continue
            ax1.plot(range(time_len),lmp_i,color = 'k')
            ax2.plot(range(time_len),dis_i,color = 'k')

        plt.xlim(0,24)
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('LMP ($/MWh)')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Dispatch (MW)')
        plt.savefig('lmp.jpg')


# def main():
    
#     metric = 'euclidean'
#     years = 6400
#     num_clusters = 50
    
#     for i in range(10):
#         dispatch_data = f'Dispatch_data_{i}.csv'
#         tsa_task = TSA64K(dispatch_data, metric, years)
#         dispatch_array = tsa_task.read_data()
#         tsa_task.read_input_pmax()
#         train_data = tsa_task.transform_data(dispatch_array)
#         labels = tsa_task.cluster_data(train_data, num_clusters, i, save_index = True)


# if __name__ == '__main__':
#     main()
