import pandas as pd
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans,silhouette_score
import matplotlib.pyplot as plt


# write the doc string before starting to code.

# This code only do clustering on dispacth power delta.

class TSA64K:
    def __init__(self, dispatch_data, method):
        # self.LMP_data = LMP_data
        self.dispatch_data = dispatch_data
        self.method = method
        # self.clusters = clusters


    def read_data(self):

        '''
        read clustering data from dispatch csv files
        
        return: 
            numpy array with dispatch data.
        '''

        df_dispatch = pd.read_csv(self.dispatch_data)

        # drop the first column
        df_dispatch = df_dispatch.iloc[: , 1:]

        # transfer the data to the np.array
        dispatch_array = df_dispatch.to_numpy(dtype = float)

        return dispatch_array

    def transform_data(self, dispatch_array):

        '''
        shape the data to the fromat that tslearn can read.

        return:
            Readable datasets for the tslearn package.
        '''
        
        datasets = []
        time_len = 24
        day_num = int(np.round(len(dispatch_array[0])/time_len))
        self.day_num = day_num

        # Test on 1 year 
        dispatch_year_0 = dispatch_array[0]

        for i in range(day_num):
            day_data = dispatch_year_0[i*time_len:(i+1)*time_len]
            datasets.append(day_data)
            
        train_data = to_time_series_dataset(datasets)
        # print(day_num)

        return train_data

    def cluster_data(self, train_data, clusters):
        '''
        cluster the data. Save the model to a json file. 

        return:
            silhouette score and label
        '''

        km = TimeSeriesKMeans(n_clusters = clusters, metric = self.method)
        labels = km.fit_predict(train_data)
        sc = silhouette_score(train_data, labels, metric = self.method)
        # print(labels)
        # print(train_data.shape)
        km.to_json('C:\\Users\\JKLCh\\Desktop\\ND_PhD\\Research\\clustering\\tslearn\\onlydispatch\\'+str(clusters)+'_clusters_onlydis.json')

        return sc, labels

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


def main():
    method = 'euclidean'

    dispatch_data = 'Example_dispatch_test.csv'

    tsa_task = TSA64K(dispatch_data, method)
    dispatch_array = tsa_task.read_data()
    train_data = tsa_task.transform_data(dispatch_array)

    tsa_task.cluster_data(train_data, 10)
    # num_range = np.concatenate([np.array([5]),np.arange(10,80,10)])
    # scores = []

    # for num in num_range:
    #     sc = tsa_task.cluster_data(train_data, num)
    #     scores.append(sc)

    # plt.plot(num_range,scores,'.',color = 'k')
    # plt.xlabel('# of clusters')
    # plt.ylabel('silhouette score_' + method)
    # plt.savefig('silhouette_score_' + method +'.jpg')

    # tsa_task.plot_origin_data(lmp_array, dispatch_array)

if __name__ == '__main__':
    main()