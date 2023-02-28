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
from clustering_dispatch_exceed_wind import ClusteringDispatchWind
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


    def _generate_label_data(self, wind_data):

        '''
        Calculate the labels for NN training. 

        Arguments:

            wind_data: the wind profile.

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        scaled_dispatch_dict = self.simulation_data._scale_data()
        sim_index = list(scaled_dispatch_dict.keys())
        single_day_dataset = {}
        dispatch_frequency_dict = {}


        for idx in sim_index:
            sim_year_data = scaled_dispatch_dict[idx]
            single_day_dataset[idx] = []
            # calculate number of days in an annual simulation
            day_num = int(len(sim_year_data)/self._time_length)
            for i, w in zip(range(day_num),wind_data):
                sim_day_data = sim_year_data[i*self._time_length:(i+1)*self._time_length]
                pem_electricity = w - sim_day_data
                dispatch_pem_data = [sim_day_data, pem_electricity]
                single_day_dataset[idx].append(dispatch_pem_data)

            to_pred_data = to_time_series_dataset(single_day_dataset[idx])
            labels = self.clustering_model.predict(to_pred_data)

            elements, count = np.unique(labels, return_counts=True)
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


    def _transform_dict_to_array_frequency(self, wind_data):

        '''
        transform the dictionary data to array that keras can train

        Arguments:
        
            None

        Returns:

            x: features (input)
            y: labels (dispatch frequency)
        '''

        dispatch_frequency_dict = self._generate_label_data(wind_data)

        index_list = list(self.simulation_data._dispatch_dict.keys())

        x = []
        y = []

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(dispatch_frequency_dict[idx])

        return np.array(x), np.array(y)


    def train_NN_frequency(self, NN_size, wind_data):

        '''
        train the dispatch frequency NN surrogate model.
        print the R2 results of each cluster.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes )

        Return:

            model: the NN model
        '''
        x, ws = self._transform_dict_to_array_frequency(wind_data)

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

        R2 = []

        if self.filter_opt == True:
            clusters = self.num_clusters + 2

        else:
            clusters = self.num_clusters

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


    def plot_R2_results(self, wind_data, NN_model_path = None, NN_param_path = None, fig_name = None):

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
            
            x, ws = self._transform_dict_to_array_frequency(wind_data)
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
                num_clusters = self.num_clusters

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
                fig.text(0.0, 0.5, 'Predicted dispatch frequency/days', va='center', rotation='vertical',font = font1)
                fig.text(0.4, 0.05, 'True dispatch frequency/days', va='center', rotation='horizontal',font = font1)
                fig.set_size_inches(10,10)

                wst = ws_test.transpose()[i]
                wsp = pred_ws_unscaled.transpose()[i]

                axs.scatter(wst*366,wsp*366,color = "green",alpha = 0.5)
                axs.plot([min(wst*366),max(wst*366)],[min(wst*366),max(wst*366)],color = "black")
                axs.set_title(f'cluster_{i}',font = font1)
                axs.annotate("$R^2 = {}$".format(round(R2[i],3)),(min(wst*366),max(wst*366)),font = font1)


                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.tick_params(direction="in",top=True, right=True)

                if fig_name == None:
                    plt.savefig(f"R2_figures\\RE_H2_plot_{self.num_clusters}clusters_cluster{i}.png",dpi =300)
                else:
                    fig_name_ = fig_name + f'_cluster_{i}'
                    plt.savefig(f"R2_figures\\{fig_name_}",dpi =300)


def main():
    num_sims = 224
    num_clusters = 20
    case_type = 'RE'
    model_type = 'frequency'
    dispatch_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Dispatch_data_RE_H2_whole.csv'
    input_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/sweep_parameters_results_RE_H2_whole.h5'
    wind_data_path = '../../../../../../datasets/results_renewable_sweep_Wind_H2/Real_Time_wind_hourly.csv'
    clustering_model_path = f'dispatch_exceed_wind_20/RE_224years_{num_clusters}clusters_Dispatch_PEM.json'

    dw = ClusteringDispatchWind(dispatch_data_path, wind_data_path, '303_WIND_1', num_sims, num_clusters)
    wind_data = dw.read_wind_data()
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    NNtrainer = TrainNNSurrogates(simulation_data, clustering_model_path, model_type, filter_opt = False)
    # dispatch_frequency_dict = NNtrainer._generate_label_data(wind_data)
    model = NNtrainer.train_NN_frequency([4,75,75,75,num_clusters],wind_data)
    NN_model_path = f'dispatch_exceed_wind_20/RE_H2_dispatch_surrogate_model_dis_pem_{num_clusters}'
    NN_param_path = f'dispatch_exceed_wind_20/RE_H2_dispatch_surrogate_param_dis_pem_{num_clusters}.json'
    # NNtrainer.save_model(model,NN_model_path,NN_param_path)
    # NNtrainer.plot_R2_results(wind_data, NN_model_path, NN_param_path)



if __name__ == '__main__':
    main()

