#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
import os

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from idaes.core.util import to_json, from_json
import numpy as np
import json
import re
import matplotlib.pyplot as plt


class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency/ revenue
    '''
    
    def __init__(self, simulation_data):

        '''
        Initialization for the class

        Arguments:

            simulation data: object, composition from ReadData class

        Return

            None
        '''

        self.simulation_data = simulation_data
        # set a class property which is the time length of a day.
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


    def calculate_capacity_factors(self):
        '''
        calculate the capacity factor of the NPP
        '''
        dispatch_data_dict, input_data_dict = self.simulation_data.read_data_to_dict()

        dispatch_cf = {}
        pem_cf = {}
        # i is the index of sweep simulations
        for i in dispatch_data_dict:
            rt_dispatch = dispatch_data_dict[i]
            rt_dispatch_cf = np.sum(rt_dispatch)/(400*len(rt_dispatch))
            dispatch_cf[i] = rt_dispatch_cf
        
        return dispatch_cf


    def _transform_dict_to_array(self):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (dispatch frequency)
        '''

        index_list = list(self.simulation_data._dispatch_dict.keys())

        x = []
        y = []


        y_dict = self.calculate_capacity_factors()

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(y_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_cf(self, NN_size):

        '''
        train the NE steady state dispatch capacity factor surrogate model.
        print the R2 results.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes)

        Return:

            model: the NN model
        '''

        x, y = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]

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
        model = keras.Sequential(name='NE_steady_state')
        model.add(layers.Input(input_layer_size))
        for layer_size in NN_size[1:-1]:
            model.add(layers.Dense(layer_size, activation='tanh'))
        model.add(layers.Dense(output_layer_size))
        model.compile(optimizer=Adam(), loss='mse')
        history = model.fit(x=x_train_scaled, y=y_train_scaled, verbose=0, epochs=500, validation_split=0.1)

        print("Making NN Predictions...") 

        # normalize the data
        x_test_scaled = (x_test - xm) / xstd
        y_test_scaled = (y_test - ym) / ystd

        print("Evaluate on test data")
        evaluate_res = model.evaluate(x_test_scaled, y_test_scaled)
        print(evaluate_res)
        print(history.history['loss'][-1])
        print(history.history['val_loss'][-1])
        predict_y = np.array(model.predict(x_test_scaled))
        predict_y_unscaled = predict_y*ystd + ym

        # calculate R2 for test data
        ypredict = predict_y_unscaled.transpose()
        SS_tot = np.sum(np.square(y_test.transpose() - ym))
        SS_res = np.sum(np.square(y_test.transpose() - ypredict))
        R2 = 1 - SS_res/SS_tot

        print('The R2 of revenue surrogate validation is ', R2)

        xmin = list(np.min(x_train_scaled,axis=0))
        xmax = list(np.max(x_train_scaled,axis=0))

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

        print('Saving model')

        # NN_model_path == none

        model.save(NN_model_path)

        with open(NN_param_path, 'w') as f:
            json.dump(self._model_params, f)

        return


    def plot_R2_results(self, NN_model_path, NN_param_path, fig_name):

        '''
        Visualize the R2 result

        Arguments: 

            train_data: list, [x, ws] where x is the input of NN and ws is output. 

            NN_model_path: the path of saved NN model

            NN_param_path: the path of saved NN params

        '''
        this_file_path = os.getcwd()

        # set the font for the plots
        font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
            }



        x, y = self._transform_dict_to_array()
        # Plot R2 for all the data
        # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        NN_model = keras.models.load_model(NN_model_path)

        with open(NN_param_path) as f:
            NN_param = json.load(f)

        # scale data
        xm = NN_param['xm_inputs']
        xstd = NN_param['xstd_inputs']
        ym = NN_param['y_mean']
        ystd = NN_param['y_std']

        x_scaled = (x - xm)/xstd
        pred_y = NN_model.predict(x_scaled)
        pred_y_unscaled = pred_y*ystd + ym

        # compute R2 over all the regression data points
        ypredict = pred_y_unscaled.transpose()
        SS_tot = np.sum(np.square(y.transpose() - ym))
        SS_res = np.sum(np.square(y.transpose() - ypredict))
        R2 = 1 - SS_res/SS_tot
        print(R2)

        # plot results.
        fig, axs = plt.subplots()
        fig.text(0.0, 0.5, 'Predicted capacity factor', va='center', rotation='vertical',font = font1)
        fig.text(0.4, 0.05, 'True capacity factor', va='center', rotation='horizontal',font = font1)
        fig.set_size_inches(10,10)

        yt = y.transpose()
        yp = pred_y_unscaled.transpose()

        axs.scatter(yt,yp,color = "green",alpha = 0.5)
        axs.plot([min(yt),max(yt)],[min(yt),max(yt)],color = "black")
        axs.set_title(f'NE Capacity Factor',font = font1)
        axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt),0.75*max(yt)),font = font1)    

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tick_params(direction="in",top=True, right=True)

        fpath = os.path.join(this_file_path,"R2_figures",f"{fig_name}")
        plt.savefig(fpath, dpi =300)
