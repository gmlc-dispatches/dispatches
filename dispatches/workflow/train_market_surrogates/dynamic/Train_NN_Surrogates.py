#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
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
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData


class TrainNNSurrogates:
    
    '''
    Train neural network surrogates for the dispatch frequency/ revenue
    '''
    
    def __init__(self, simulation_data, data_file, filter_opt = True):

        '''
        Initialization for the class

        Arguments:

            simulation data: object, composition from ReadData class

            data_file: path of the data file. If the model_type = frequency, the data_file should be the clustering_model_path, 
            if the model_type = revenue, the data_file should be the revenue.csv

            filter_opt: bool, if we are going to filter out 0/1 capacity days

        Return

            None
        '''

        self.simulation_data = simulation_data
        self.data_file = data_file
        self.filter_opt = filter_opt
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
        
        if not isinstance(value, SimulationData):
            raise TypeError(
                f"The simulation_data must be created from SimulationData."
            )
        self._simulation_data = value


    @property
    def data_file(self):

        '''
        Porperty getter of data_file

        Arguments:

            None

        Returns:

            data_file
        '''
        
        return self._data_file


    @data_file.setter
    def data_file(self, value):

        '''
        Porperty setter of data_file
        
        Arguments:

            value: str, path of the clustering model

        Returns:

            None
        '''
        
        if not (isinstance(value, str) or isinstance(value, pathlib.WindowsPath) or isinstance(value, pathlib.PosixPath)):
            raise TypeError(
                f"The data_file must be str or object, but {type(value)} is given."
            )
        self._data_file = value


    @property
    def filter_opt(self):

        '''
        Property getter of filter_opt

        Arguments:
        
            None

        Return:
        
            bool: if want filter 0/1 days in clustering
        '''

        return self._filter_opt


    @filter_opt.setter
    def filter_opt(self, value):

        '''
        Property setter of filter_opt

        Arguments:
        
            value: bool.
        
        Returns:
        
            None
        '''

        if not isinstance(value, bool):
            raise TypeError(
                f"Filter_opt must be bool, but {type(value)} is given"
            )

        self._filter_opt = value



    def _read_clustering_model(self, clustering_model_path):

        '''
        Read the time series clustering model from the given path

        Arguments:

            clustering_model_path: path of clustering model

        Returns:

            Clustering model
        '''

        clustering_model = TimeSeriesKMeans.from_json(clustering_model_path)

        # read the number of clusters from the clustering model
        self.num_clusters = clustering_model.n_clusters
        self.clustering_model = clustering_model

        return clustering_model


    def _generate_label_data(self):

        '''
        Calculate the labels for NN training. 

        Arguments:

            None

        Return:

            dispatch_frequency_dict: {run_index: [dispatch frequency]}

        '''
        # scale the dispatch data
        scaled_dispatch_dict = self.simulation_data._scale_data()
        sim_index = list(scaled_dispatch_dict.keys())
        single_day_dataset = {}
        dispatch_frequency_dict = {}
        
        # filter out 0/1 days in each simulaion data
        if self.filter_opt == True:          
            for idx in sim_index:
                sim_year_data = scaled_dispatch_dict[idx]
                single_day_dataset[idx] = []
                # calculate number of days in a simulation
                day_num = int(len(sim_year_data)/self._time_length)
                zero_day = 0
                full_day = 0
                
                for day in range(day_num):
                    # slice the annual data into days
                    sim_day_data = sim_year_data[day*self._time_length:(day+1)*self._time_length]
                    
                    if sum(sim_day_data) == 0:
                        zero_day += 1
                    
                    elif sum(sim_day_data) == 24:
                        full_day += 1
                   
                    else:
                        single_day_dataset[idx].append(sim_day_data)
            
                # frequency of 0/1 days
                ws0 = zero_day/day_num
                ws1 = full_day/day_num


                if len(single_day_dataset[idx]) == 0:
                    labels = np.array([])

                else:
                    to_pred_data = to_time_series_dataset(single_day_dataset[idx])
                    labels = self.clustering_model.predict(to_pred_data)

                # count the how many representative days and how many days in the representative days
                elements, count = np.unique(labels,return_counts=True)

                pred_result_dict = dict(zip(elements, count))
                count_dict = {}

                for j in range(self.num_clusters):
                    
                    if j in pred_result_dict.keys():
                        # if there are days in this simulation year belong to cluster i, count the frequency 
                        count_dict[j] = pred_result_dict[j]/day_num
                    
                    else:
                        # else, the frequency of this cluster is 0
                        count_dict[j] = 0

                # the first element in w is frequency of 0 cf days
                dispatch_frequency_dict[idx] = [ws0]

                for key, value in count_dict.items():
                    dispatch_frequency_dict[idx].append(value)

                # the last element in w is frequency of 1 cf days
                dispatch_frequency_dict[idx].append(ws1)
        
        # filter_opt = False then we do not filter 0/1 days
        else:
            for idx in sim_index:
                sim_year_data = scaled_dispatch_dict[idx]
                single_day_dataset[idx] = []
                # calculate number of days in a simulation
                day_num = int(len(sim_year_data)/self._time_length)
                
                for day in range(day_num):
                    sim_day_data = sim_year_data[day*self._time_length:(day+1)*self._time_length]
                    single_day_dataset[idx].append(sim_day_data)

                to_pred_data = to_time_series_dataset(single_day_dataset[idx])
                labels = self.clustering_model.predict(to_pred_data)

                elements, count = np.unique(labels,return_counts=True)
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

        if self.model_type == 'frequency':
            y_dict = self._generate_label_data()

        if self.model_type == 'revenue':
            y_dict = self.simulation_data.read_rev_data(self.data_file)

        for idx in index_list:
            x.append(self.simulation_data._input_data_dict[idx])
            y.append(y_dict[idx])

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
        # set the class property model_type
        self.model_type = 'frequency'

        # read and save the clustering model in self.clustering_model
        self._read_clustering_model(self.data_file)

        x, ws = self._transform_dict_to_array()

        # the first element of the NN_size dict is the input layer size, the last element is output layer size. 
        input_layer_size = NN_size[0]
        output_layer_size = NN_size[-1]

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
        # excpt the first and last element in the list are the hidden layer size.
        for layer_size in NN_size[1:-1]:
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

        if self.filter_opt == True:
            clusters = self.num_clusters + 2

        else:
            clusters = self.num_clusters

        R2 = []

        for rd in range(0,clusters):
            # compute R2 metric
            wspredict = predict_ws_unscaled.transpose()[rd]
            SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
            SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
            residual = 1 - SS_res/SS_tot
            R2.append(residual)

        print('The R2 of dispatch surrogate validation is', R2)

        xmin = list(np.min(x_train_scaled, axis=0))
        xmax = list(np.max(x_train_scaled, axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
            "ws_mean":list(wsm),"ws_std":list(wsstd)}

        self._model_params = data

        return model


    def train_NN_revenue(self, NN_size):

        '''
        train the revenue NN surrogate model.
        print the R2 results.

        Arguments:
            
            NN_size: list, the size of neural network. (input nodes, hidden layer 1 size, ..., output nodes)

        Return:

            model: the NN model
        '''

        self.model_type = 'revenue'
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
        model = keras.Sequential(name=self.model_type)
        model.add(layers.Input(input_layer_size))
        for layer_size in NN_size[1:-1]:
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

        xmin = list(np.min(x_train_scaled,axis=0))
        xmax = list(np.max(x_train_scaled,axis=0))

        data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax, "y_mean":ym,"y_std":ystd}

        self._model_params = data

        return model


    def save_model(self, model, NN_model_path = None, NN_param_path = None):

        '''
        Save the model to the path which can be specified by the user. 

        Arguments:

            model: trained model from self.train_NN()

            NN_model_fpath: the keras NN model path, if fpath == None, save to default path. 

            NN_param_path: the NN scaling parameter path, if fpath == None, save to default path. 

        Return:

            path_list: list of model and param path
        '''

        print('Saving model')

        # NN_model_path == none
        if NN_model_path == None:
            # save the NN model
            model_path = pathlib.Path(f'{self.simulation_data.case_type}_{self.model_type}_NN_model')
            folder_path = pathlib.Path(f'{self.simulation_data.case_type}_case_study')
            model_save_path = pathlib.Path.cwd().joinpath(folder_path,model_path)
            model.save(model_save_path)

        else: 
            model_save_path = str(pathlib.Path(NN_model_path).absolute())
            model.save(model_save_path)

        if NN_param_path == None:
            # save the sacling parameters to current path
            param_path = pathlib.Path(f'{self.simulation_data.case_type}_{self.model_type}NN_params.json')
            folder_path = pathlib.Path(f'{self.simulation_data.case_type}_case_study')
            param_save_path = pathlib.Path.cwd().joinpath(folder_path,param_path)

            with open(param_save_path, 'w') as f:
                json.dump(self._model_params, f)
        else:
            param_save_path = str(pathlib.Path(NN_param_path).absolute())
            with open(param_save_path, 'w') as f:
                json.dump(self._model_params, f)

        path_list = [model_save_path, param_save_path]
        
        return path_list 


    def plot_R2_results(self, NN_model_path, NN_param_path, fig_name = None):

        '''
        Visualize the R2 result

        Arguments: 

            NN_model_path: the path of saved NN model

            NN_param_path: the path of saved NN params

            fig_name: path of the figure folder, if None, save to default path.

        Returns:

            None
        '''
        # set the font for the plots
        font1 = {'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 18,
            }

        if self.model_type == 'frequency':
            
            x, ws = self._transform_dict_to_array()
            # use a different random_state from the training
            x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=0)

            # load the NN model from the given path, make the path an absoulte path
            model_save_path = str(pathlib.Path(NN_model_path).absolute())
            NN_model = keras.models.load_model(model_save_path)


            # load the NN parameters from the given path, make the path an absoulte path
            param_save_path = str(pathlib.Path(NN_param_path).absolute())
            with open(param_save_path) as f:
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
                num_clusters = self.num_cluster

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
                fig.text(0.0, 0.5, 'Predicted dispatch frequency', va='center', rotation='vertical',font = font1)
                fig.text(0.4, 0.05, 'True dispatch frequency', va='center', rotation='horizontal',font = font1)
                fig.set_size_inches(10,10)

                wst = ws_test.transpose()[i]
                wsp = pred_ws_unscaled.transpose()[i]

                axs.scatter(wst*366,wsp*366,color = "green",alpha = 0.5)
                # plot by day instead of frequency
                axs.plot([min(wst)*366,max(wst)*366],[min(wst)*366,max(wst)*366],color = "black")
                # axs.set_xlim(-5,370)
                # axs.set_ylim(-5,370)
                axs.set_title(f'cluster_{i}',font = font1)
                axs.annotate("$R^2 = {}$".format(round(R2[i],3)),(min(wst)*366,0.75*max(wst)*366),font = font1)


                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.tick_params(direction="in",top=True, right=True)

                if fig_name == None:
                    folder_path = f'{self.case_type}_case_study/R2_figures'
                    if not os.path.isdir(folder_path):
                        os.mkdir(folder_path)
                    default_path = str(pathlib.Path.cwd().joinpath(folder_path, f'{self.simulation_data.case_type}_dispatch_cluster{i}.png'))
                    plt.savefig(default_path, dpi =300)
                else:
                    folder_path = pathlib.Path(fig_name).absolute()
                    fig_name = pathlib.Path(f'{self.simulation_data.case_type}_dispatch_cluster{i}.png')
                    fpath = folder_path/fig_name
                    plt.savefig(fpath, dpi =300)


        if self.model_type == 'revenue':

            x, y = self._transform_dict_to_array()
            # use a different random_state from the training
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

            # load the NN model
            model_save_path = str(pathlib.Path(NN_model_path).absolute())
            NN_model = keras.models.load_model(model_save_path)

            # load the NN parameters from the given path, make the path an absoulte path
            param_save_path = str(pathlib.Path(NN_param_path).absolute())
            with open( param_save_path) as f:
                NN_param = json.load(f)

            # scale data
            xm = NN_param['xm_inputs']
            xstd = NN_param['xstd_inputs']
            ym = NN_param['y_mean']
            ystd = NN_param['y_std']

            x_test_scaled = (x_test - xm)/xstd
            pred_y = NN_model.predict(x_test_scaled)
            pred_y_unscaled = pred_y*ystd + ym

            # compute R2
            ypredict = pred_y_unscaled.transpose()
            SS_tot = np.sum(np.square(y_test.transpose() - ym))
            SS_res = np.sum(np.square(y_test.transpose() - ypredict))
            R2 = 1 - SS_res/SS_tot
            print(R2)

            # plot results.
            fig, axs = plt.subplots()
            fig.text(0.0, 0.5, 'Predicted revenue/$', va='center', rotation='vertical',font = font1)
            fig.text(0.4, 0.05, 'True revenue/$', va='center', rotation='horizontal',font = font1)
            fig.set_size_inches(10,10)

            yt = y_test.transpose()
            yp = pred_y_unscaled.transpose()

            axs.scatter(yt,yp,color = "green",alpha = 0.5)
            axs.plot([min(yt),max(yt)],[min(yt),max(yt)],color = "black")
            axs.set_title(f'Revenue',font = font1)
            axs.annotate("$R^2 = {}$".format(round(R2,3)),(min(yt),0.75*max(yt)),font = font1)    

            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.tick_params(direction="in",top=True, right=True)

            if fig_name == None:
                folder_path = f'{self.case_type}_case_study/R2_figures'
                if not os.path.isdir(folder_path):
                    os.mkdir(folder_path)
                default_path = str(pathlib.Path.cwd().joinpath(folder_path, f'{self.simulation_data.case_type}_revenue.png'))
                plt.savefig(default_path, dpi =300)
            
            else:
                fpath = str(pathlib.Path.cwd().joinpath(f'{self.simulation_data.case_type}','R2_figures',f'{fig_name}'))
                folder_path = pathlib.Path(fig_name).absolute()
                fig_name = pathlib.Path(f'{self.simulation_data.case_type}_revenue.png')
                fpath = folder_path/fig_name
                plt.savefig(fpath, dpi =300)
