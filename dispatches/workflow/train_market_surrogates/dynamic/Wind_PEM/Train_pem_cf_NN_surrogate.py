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
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from clustering_dispatch_pem_cf_wind import ClusteringDispatchWind
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os



# use the clustering result for dispatch+wind from wind_h2 sweep

# input layer 4 nodes, output 30 nodes for frequency of each representative day.

# attention: this works for wind+dispatch 2-d data.

class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency

    For RE case, filter is False.
    '''
    
    def __init__(self, simulation_data):

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
        self._time_length = 24
        

    def _generate_label_data(self, real_pem_elec_cf):

        '''
        Calculate the labels for NN training. 

        Arguments:

            wind_data: the wind profile.

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        pem_cf_dict = {}

        for i in range(len(real_pem_elec_cf)):
            # year average cf.
            pem_cf_dict[i] = sum(real_pem_elec_cf[i])/24/366
                
        return pem_cf_dict


    def _transform_dict_to_array(self, real_pem_elec_cf):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (dispatch frequency)
        '''

        pem_cf_dict = self._generate_label_data(real_pem_elec_cf)

        index_list = list(self.simulation_data._dispatch_dict.keys())

        x = []
        y = []

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(pem_cf_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_pem_cf(self, NN_size, real_pem_elec_cf):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''
        x, y = self._transform_dict_to_array(real_pem_elec_cf)

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        del NN_size[0]
        del NN_size[-1]

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

        # scale the data both x and ws
        xm = np.mean(x_train,axis = 0)
        xstd = np.std(x_train,axis = 0)
        ym = np.mean(y_train,axis = 0)
        ystd = np.std(y_train,axis = 0)
        x_train_scaled = (x_train - xm) / xstd
        y_train_scaled = (y_train - ym)/ ystd

        # train a keras MLP (multi-layer perceptron) Regressor model
        model = keras.Sequential(name='pem_annual_average_cf')
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

        print('The R2 of revenue surrogate validation is ', R2)

        print(R2)

        xmin = list(np.min(x_train_scaled, axis=0))
        xmax = list(np.max(x_train_scaled, axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax, "y_mean":ym,"y_std":ystd}
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

        model.save(NN_model_path)
        with open(NN_param_path, 'w') as f:
            json.dump(self._model_params, f)

        return

    def plot_R2_results(self, real_pem_elec_cf, NN_model_path, NN_param_path):
            # set the font for the plots
            font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18}

            x, y = self._transform_dict_to_array(real_pem_elec_cf)

            # Load NN model and scaling params
            NN_model = keras.models.load_model(NN_model_path)
            
            with open(NN_param_path) as f:
                NN_param = json.load(f)

            # scale data
            xm = NN_param['xm_inputs']
            xstd = NN_param['xstd_inputs']
            ym = NN_param['y_mean']
            ystd = NN_param['y_std']

            x_test_scaled = (x- xm)/xstd
            pred_y = NN_model.predict(x_test_scaled)
            pred_y_unscaled = pred_y*ystd + ym

            # compute R2
            ypredict = pred_y_unscaled.transpose()
            SS_tot = np.sum(np.square(y.transpose() - ym))
            SS_res = np.sum(np.square(y.transpose() - ypredict))
            R2 = 1 - SS_res/SS_tot
            print(R2)

            # plot results.
            fig, axs = plt.subplots()
            fig.text(0.0, 0.5, 'Predicted revenue/$', va='center', rotation='vertical',font = font1)
            fig.text(0.4, 0.05, 'True revenue/$', va='center', rotation='horizontal',font = font1)
            fig.set_size_inches(10,10)

            h2_price = 3
            h2_conversion = 54.953
            yt = y.transpose()
            yp = pred_y_unscaled.transpose()
            for i in range(len(yt)):
                pem_pmax = self.simulation_data._input_data_dict[i][1]
                yt[i] = yt[i]*pem_pmax*366*24/h2_conversion*h2_price/1e3
                yp[0][i] = yp[0][i]*pem_pmax*366*24/h2_conversion*h2_price/1e3
            
            axs.scatter(yt,yp,color = "green",alpha = 0.5)
            axs.plot([min(yt),max(yt)],[min(yt),max(yt)],color = "black")
            axs.set_title(f'H2 Revenue surrogate/M$',font = font1)
            axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt),0.75*max(yt)),font = font1)    

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)

            fpath = 'RE_H2_Rev_surrogate.jpg'
            plt.savefig(fpath, dpi =300)

def main():
    num_sims = 224
    num_clusters = 20
    case_type = 'RE'
    model_type = 'frequency'
    dispatch_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_whole.csv'
    input_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    wind_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Real_Time_wind_hourly.csv'

    dw = ClusteringDispatchWind(dispatch_data_path, input_data_path, wind_data_path, '303_WIND_1', num_sims, num_clusters)
    dispatch_array = dw.read_data()
    real_pem_elec_cf = dw.calculate_PEM_cf(dispatch_array)
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    NNtrainer = TrainNNSurrogates(simulation_data)
    # dispatch_frequency_dict = NNtrainer._generate_label_data(real_pem_elec_cf)
    # model = NNtrainer.train_NN_pem_cf([4,75,75,75,1],real_pem_elec_cf)
    NN_model_path = f'PEM_H2_REVENUE_surrogate/RE_H2_pem_cf_only_surrogate_model'
    NN_param_path = f'PEM_H2_REVENUE_surrogate/RE_H2_pem_cf_only_surrogate_param.json'
    # NNtrainer.save_model(model,NN_model_path,NN_param_path)
    NNtrainer.plot_R2_results(real_pem_elec_cf, NN_model_path, NN_param_path)



if __name__ == '__main__':
    main()
