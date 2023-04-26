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
    
    def __init__(self, simulation_data, NPV_data_path):

        '''
        Initialization for the class

        Arguments:

            NPV_data_path: str, path for sweep data

        Return

            None
        '''
        self.simulation_data = simulation_data
        self.NPV_data_path = NPV_data_path


    def _read_NPV_data(self):

        # read NPV data
        df_NPV = pd.read_csv(self.NPV_data_path)
        NPV_array = df_NPV.iloc[:, 1:].to_numpy(dtype = float)
        
        return NPV_array


    def _generate_label_data(self):

        '''
        Calculate the labels for NN training. 

        Arguments:

            wind_data: the wind profile.

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        NPV_array = self._read_NPV_data()
        NPV_dict = {}
        for i in range(len(NPV_array)):
            NPV_dict[i] = NPV_array[i][0]

        return NPV_dict


    def _transform_dict_to_array(self):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (dispatch frequency)
        '''

        NPV_dict = self._generate_label_data()

        index_list = list(NPV_dict.keys())

        x = []
        y = []

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(NPV_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_NPV(self, NN_size):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''
        x, y = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]
        del NN_size[0]
        del NN_size[-1]

        # train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=0)

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
        history = model.fit(x=x_train_scaled, y=y_train_scaled, verbose=0, epochs=500, validation_split = 0.2)

        print("Making NN Predictions...") 

        # normalize the data
        x_test_scaled = (x_test - xm) / xstd
        y_test_scaled = (y_test - ym) / ystd

        print("Evaluate on test data")
        # evaluate_res = model.evaluate(x_test_scaled, y_test_scaled)
        # print(evaluate_res)
        print(history.history['loss'][-1])
        print(history.history['val_loss'][-1])
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

        x, y = self._transform_dict_to_array()

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
        fig.text(0.0, 0.5, 'Predicted NPV/$', va='center', rotation='vertical',font = font1)
        fig.text(0.4, 0.05, 'True NPV/$', va='center', rotation='horizontal',font = font1)
        fig.set_size_inches(10,10)

        yt = y.transpose()
        yp = pred_y_unscaled.transpose()
        
        axs.scatter(yt,yp,color = "green",alpha = 0.5)
        axs.plot([min(yt),max(yt)],[min(yt),max(yt)],color = "black")
        axs.set_title(f'RE NPV surrogate/M$',font = font1)
        axs.annotate("$R^2 = {}$".format(round(R2,3)),(-1.8e9,-1.1e9),font = font1)    

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tick_params(direction="in",top=True, right=True)

        fpath = 'RE_H2_NPV_surrogate.jpg'
        plt.savefig(fpath, dpi =300)

        pem_bid = np.array([15,20,25,30,35,40,45])
        pem_power = np.array([127.5,169.4,211.75,254.1,296.45,338.8,381.15,423.5])
        result_dict = {}
        rf_max_lmp_pair = [(10,500),(10,1000),(15,500),(15,1000)]
        c = 0
        for p in rf_max_lmp_pair:
            ratio_arrray = np.zeros((len(pem_power),len(pem_bid)))
            surrogate_NPV_array = np.zeros((len(pem_power),len(pem_bid)))
            sweep_NPV_array = np.zeros((len(pem_power),len(pem_bid)))
            for i in range(len(pem_power)):
                for j in range(len(pem_bid)):
                    r = yt[c]/yp[0][c]
                    ratio_arrray[i][j] = r
                    surrogate_NPV_array[i][j] = yp[0][c]/1e6     # M USD
                    sweep_NPV_array[i][j] = yt[c]/1e6      # M USD
                    c += 1
            result_dict[p] = [ratio_arrray, surrogate_NPV_array, sweep_NPV_array]
        
        for p in result_dict:
            fig, axs = plt.subplots(1,3, figsize =(16,9))
            title = ['surrogate_NPV/sweep_NPV', 'surrogate_NPV', 'sweep_NPV']
            for m in range(len(axs)):
                im = axs[m].imshow(result_dict[p][m].T, origin='lower')

                # Show all ticks and label them with the respective list entries
                axs[m].set_xticks(np.arange(len(pem_power)), labels=pem_power)
                axs[m].set_yticks(np.arange(len(pem_bid)), labels=pem_bid)
                axs[m].set_xlabel('pem power/MW')
                axs[m].set_ylabel('pem bid/$')
                # Rotate the tick labels and set their alignment.
                plt.setp(axs[m].get_xticklabels(), rotation=45, ha="right",
                            rotation_mode="anchor")

                # Loop over data dimensions and create text annotations.
                for i in range(len(pem_power)):
                    for j in range(len(pem_bid)):
                        if m == 0:
                            text = axs[m].text(i, j, np.round(result_dict[p][m][i, j],3),
                                            ha="center", va="center", color="r")
                            axs[m].set_title(f"RE" + title[m] + f" ({p[0]}, {p[1]})")
                        else:
                            text = axs[m].text(i, j, np.round(result_dict[p][m][i, j],1),
                                            ha="center", va="center", color="r")

                            axs[m].set_title(f"RE" + title[m] + f" ({p[0]}, {p[1]}), M$")
                                
            fig.tight_layout()
            plt.savefig(f'RE NPV {p[0],p[1]}', dpi =300)


def main():
    num_sims = 224
    case_type = 'RE'
    wind_gen = '303_WIND_1'
    dispatch_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_whole.csv'
    input_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    wind_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Real_Time_wind_hourly.csv'
    NPV_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/RE_NPV.csv'
    
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    NNtrainer = TrainNNSurrogates(simulation_data, NPV_data_path)

    model = NNtrainer.train_NN_NPV([4,25,25,1])

    NN_model_path = f'NPV_surrogate_model'
    NN_param_path = f'npv_surrogate_param.json'
    NNtrainer.save_model(model,NN_model_path,NN_param_path)
    NNtrainer.plot_R2_results(NN_model_path, NN_param_path)



if __name__ == '__main__':
    main()
