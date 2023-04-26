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

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clustering_dispatch_wind_pem_static import ClusteringDispatchWind
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os
import pickle

'''
This script trains NN frequenct surrogate using the static clustering model 

for (P_grid, P_pem, P_wind).
'''

class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency

    For RE case, filter is False.
    '''
    
    def __init__(self, simulation_data, clustering_class, clustering_model_path):

        '''
        Initialization for the class

        Arguments:
            simulation data: object, composition from ReadData class

            clustering_model_path: path of the saved clustering model

        Return

            None
        '''
        self.simulation_data = simulation_data
        self.clustering_class = clustering_class
        self.clustering_model_path = clustering_model_path
        

    def _read_clustering_model(self):
        # read clustering model
        with open (self.clustering_model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model


    def _generate_label_data(self):

        '''
        Calculate the labels for NN training. 

        Arguments:

            wind_data: the wind profile.

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        train_data = self.clustering_class.transform_data()
        clustering_model = self._read_clustering_model()
        
        # for train_data, the shape is 196716*3. Each 366*24 hour data is one year
        num_sims = int(len(train_data)/366/24)
        total_year_hours = 8784     # 366*24
        
        # reshape data into (num_sims), so that we can visit the data by each sweep simulation.
        label_data_reshaped = clustering_model.labels_.reshape((num_sims, total_year_hours))

        dispatch_frequency_dict = {}

        for idx in range(num_sims):
            # year_data has shape of (8784,3)
            elements, count = np.unique(label_data_reshaped[idx], return_counts=True)
            pred_result_dict = dict(zip(elements, count))
            count_dict = {}

            for j in range(self.clustering_class.num_clusters):

                if j in pred_result_dict.keys():
                    count_dict[j] = pred_result_dict[j]/total_year_hours
                
                else:   # if the frequency of cluster x is 0 in some years
                    count_dict[j] = 0

            dispatch_frequency_dict[idx] = []

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


    def train_NN_frequency(self, NN_size):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''
        x, ws = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        del NN_size[0]
        del NN_size[-1]

        # train test split
        x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=0)

        # scale the data both x and ws
        xm = np.mean(x_train,axis = 0)
        xstd = np.std(x_train,axis = 0)
        wsm = np.mean(ws_train,axis = 0)
        wsstd = np.std(ws_train,axis = 0)
        x_train_scaled = (x_train - xm) / xstd
        ws_train_scaled = (ws_train - wsm)/ wsstd

        # train a keras MLP (multi-layer perceptron) Regressor model
        model = keras.Sequential(name='static_clustering_NN')
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
        print(history.history['loss'][-1])
        predict_ws = np.array(model.predict(x_test_scaled))
        predict_ws_unscaled = predict_ws*wsstd + wsm

        R2 = []

        for rd in range(self.clustering_class.num_clusters):
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


    def save_model(self, model, NN_model_path, NN_param_path):

        '''
        Save the model to the path which can be specified by the user. 

        Arguments:

            model: trained model from self.train_NN()

            fpath: if fpath == None, save to default path. 

        Return:

            None
        '''
        # save the NN model
        model.save(NN_model_path)

        # save scaling parameters
        with open(NN_param_path, 'w') as f:
            json.dump(self._model_params, f)

        return


    def plot_R2_results(self, NN_model_path, NN_param_path):
        
        # set the font for the plots
        font1 = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
        
        x, ws = self._transform_dict_to_array()

        # load the NN model from the given path
        NN_model = keras.models.load_model(NN_model_path)

        with open(NN_param_path, 'r') as f:
            NN_param = json.load(f)

        # scale data
        xm = NN_param['xm_inputs']
        xstd = NN_param['xstd_inputs']
        wsm = NN_param['ws_mean']
        wsstd = NN_param['ws_std']

        x_test_scaled = (x - xm)/xstd
        pred_ws = NN_model.predict(x_test_scaled)
        pred_ws_unscaled = pred_ws*wsstd + wsm

        # calculate the R2 for each representative day
        R2 = []

        for rd in range(self.clustering_class.num_clusters):
            # compute R2 metric
            wspredict = pred_ws_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws.transpose()[rd] - wspredict))
            residual = 1 - SS_res/SS_tot
            R2.append(residual)
        
        print(R2)
        
        # plot the figure
        for i in range(self.clustering_class.num_clusters):
            fig, axs = plt.subplots()
            fig.text(0.0, 0.5, 'Predicted dispatch frequency/hours', va='center', rotation='vertical',font = font1)
            fig.text(0.4, 0.05, 'True dispatch frequency/hours', va='center', rotation='horizontal',font = font1)
            fig.set_size_inches(10,10)

            wst = ws.transpose()[i]
            wsp = pred_ws_unscaled.transpose()[i]

            axs.scatter(wst*366*24,wsp*366*24,color = "green",alpha = 0.5)
            axs.plot([min(wst*366*24),max(wst*366*24)],[min(wst*366*24),max(wst*366*24)],color = "black")
            axs.set_title(f'cluster_{i}',font = font1)
            axs.annotate("$R^2 = {}$".format(round(R2[i],3)),(min(wst*366*24),max(wst*366*24)),font = font1)


            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)

            fig_name_ = f'static_clustering_NN_cluster_{i}.jpg'
            plt.savefig(f"R2_figures\\{fig_name_}",dpi =300)
    
    

    def check_results(self, input_data_path):
        # read the dispatch data
        dispatch_array = self.clustering_class.read_dispatch_data()
        wind_data = self.clustering_class.read_wind_data()
        pem_data = self.clustering_class.calculate_PEM_cf()

        df_input_data = pd.read_hdf(input_data_path)
        num_col = df_input_data.shape[1]
        num_row = df_input_data.shape[0]
        X = df_input_data.iloc[list(range(num_row)),list(range(1,num_col))].to_numpy()

        clustering_model = self._read_clustering_model()
        dispatch_frequency_dict = self._generate_label_data()

        check_dict = {}
        true_dict = {}
        for i in dispatch_frequency_dict:
            d = 0
            p = 0
            w = 0
            for j in range(len(clustering_model.cluster_centers_)):
                d += dispatch_frequency_dict[i][j]*clustering_model.cluster_centers_[j][0]    # dispatch
                p += dispatch_frequency_dict[i][j]*clustering_model.cluster_centers_[j][1]    # PEM
                w += dispatch_frequency_dict[i][j]*clustering_model.cluster_centers_[j][2]    # wind
            rd = sum(dispatch_array[i]/847)/8784
            rp = sum(pem_data[i])/8784
            rw = sum(wind_data/847)/8784
            check_dict[i] = [d,p,w]
            true_dict[i] = [rd,rp,rw]

        for i, j in zip(check_dict, true_dict):
            print(true_dict[j], check_dict[i])

        return

def main():
    num_sims = 224
    num_clusters = 20
    case_type = 'RE'
    wind_gen = '303_WIND_1'
    dispatch_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_whole.csv'
    input_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    wind_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Real_Time_wind_hourly.csv'
    clustering_model_path = 'static_clustering_wind_pmax.pkl'
    
    # clustering_class = ClusteringDispatchWind(dispatch_data_path, wind_data_path, input_data_path, wind_gen, num_clusters)
    # simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    # NNtrainer = TrainNNSurrogates(simulation_data, clustering_class, clustering_model_path)

    # dispatch_frequency_dict = NNtrainer._generate_label_data()
    with open(clustering_model_path, 'rb') as f:
        model = pickle.load(f)
    for i in model.cluster_centers_:
        print(i[0] + i[1] - i[2])
    # model = NNtrainer.train_NN_frequency([4,45,75,45,num_clusters])
    # NN_model_path = f'steady_state/ss_surrogate_model_wind_pmax'
    # NN_param_path = f'steady_state/ss_surrogate_param_wind_pmax.json'
    # NNtrainer.save_model(model,NN_model_path,NN_param_path)
    # NNtrainer.plot_R2_results(NN_model_path, NN_param_path)



if __name__ == '__main__':
    main()
