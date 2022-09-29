import os

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from Time_series_clustering.only_dispatch.filter_01_6400_years import TSA64K
from Time_series_clustering.train_kerasNN.TSA_NN_surrogate_keras import load_cluster_model, calculate_ws, read_input_x, train_NN_surrogate
# from conceptual_design_dynamic.new_full_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE, record_result
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from pyomo.environ import value, SolverFactory
from idaes.core.util import to_json, from_json
import time
import pandas as pd
import numpy as np
import json
import re


# need to find out a place to store the data instead of just put them in the dispatches repo
# temporarily put them in the repo


class ReadData:
	def __init__(self, num_sims = 1):

		# How many simulation years data we would like to read.
		self.num_sims = num_sims

	@property
	def num_sims(self):

		'''
		Porperty getter of num_sims

		Returns:

			int: number of years for clustering, positive integer.
		'''

		return self._num_sims


	@num_sims.setter
	def num_sims(self, value):

		'''
		Property setter of num_sims

		Arguments:

			value: intended value for num_sims 

		Returns:
			
			None
		'''

		if not isinstance(value, int):
			raise TypeError(
				f"The number of clustering years must be positive integer, but {type(value)} is given."
			)

		if value < 1:
			raise ValueError(
				f"The number of simulation years must be positive integer, but {value} is given."
			)

		self._num_sims = value


	def _read_data_to_array(self, dispatch_data_file, input_data_file):

		'''
		Read the dispatch data from the csv file

		Arguments:
			
			dispatch_data_fil: the file stores dispatch profiles by simulation years

			input_data_file: the file stores input data for parameter sweep

		Returns:
			
			numpy.ndarray: dispatch data
		'''

		df_dispatch = pd.read_csv(dispatch_data_file, nrows = self.num_sims)

		# drop the first column
		df_dispatch_data = df_dispatch.iloc[: , 1:]

		# the first column is the run_index. Put them in an array
		df_index = df_dispatch.iloc[:,0]
		run_index = df_index.to_numpy(dtype = str)

		# save the index in an list.
		index = []
		for run in run_index:
		    index_num = re.split('_|\.',run)[1]
		    index.append(int(index_num))

		# transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
		dispatch_array = df_dispatch_data.to_numpy(dtype = float)

		return dispatch_array, index


	def read_data_to_dict(self, dispatch_data_file, input_data_file):
		
		'''
		Transfer the data into dictionary 
		
		Arguments: 
			
			dispatch_data_file: the file stores dispatch profiles by simulation years

			input_data_file: the file stores input data for parameter sweep

		Returns:
			
			dispatch_dict: {run_index:[dispatch data]}

			input_dict: {run_index:[input data]}
		'''

		dispatch_array, index = self._read_data_to_array(dispatch_data_file, input_data_file)

		dispatch_dict = {}

		for num, idx in enumerate(index):
			dispatch_dict[idx] = dispatch_array[num]

		sim_index = list(dispatch_dict.keys())

		# read the input data
		df_input_data = pd.read_hdf(input_data_file)

		X = df_input_data.iloc[sim_index,[1,2,3,4,5,6,7,9]].to_numpy()

		input_dict = {}

		for num, x in zip(sim_index, X):
			input_dict[num] = x

		return dispatch_dict, input_dict



class TimeSeriesClustering(ReadData):
	
	'''
	Inheret from ReadData so that we can read as many data as we want for clustering.

	Time series clustering for the dispatch data. 

	Now only can do clustering over dispatch data.

	'''

	def __init__(self, num_sims, num_clusters, filter_opt = True, metric = 'euclidean'):

		super(TimeSeriesClustering, self).__init__(num_sims)
		self.metric = metric
		self.num_clusters = num_clusters
		self.filter_opt = filter_opt
		self._time_length = 24


	@property
	def metric(self):

		'''
		Porperty getter of metric

		Returns:
			metric
		'''

		return self._metric


	@metric.setter
	def metric(self, value):

		'''
		Property setter for metric

		Returns:
	        None
        '''

		if not (value == 'euclidean' or value == 'dtw'): 
			raise ValueError(
				f"The metric must be one of euclidean or dtw, but {value} is provided"
			)
		
		self._metric = value


	@property
	def num_clusters(self):

		'''
		Property getter of num_clusters

		Returns:
			int: number of clusters for the clustering
			(k-means need given number of clusters)
		'''

		return self._num_clusters

    
	@num_clusters.setter
	def num_clusters(self, value):

		'''
		Property setter of num_clusters

		Returns:
			None
		'''

		if not isinstance(value, int):
			raise TypeError(
				f"Number of clusters must be integer, but {type(value)} is given"
		 	)

		self._num_clusters = value


	@property
	def filter_opt(self):

		'''
		Property getter of filter_opt

		Return:
			bool: if want filter 0/1 days in clustering
        '''
		return self._filter_opt


	@filter_opt.setter
	def filter_opt(self, value):

		'''
		Property setter of filter_opt

		Returns:
			None
		'''

		if not isinstance(value, bool):
			raise TypeError(
				f"filter_opt must be bool, but {type(value)} is given"
			)

		self._filter_opt = value


	def _read_pmax(self, dispatch_dict, input_data_dict):

		'''
		Read pmax from input_dict
		
		Arguments:
			
			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		
		Returns:
			pmax_dict: {run_index: pmax}
		'''

		index_list = list(dispatch_dict.keys())

		pmax_dict = {}

		for idx in index_list:
			pmax = input_data_dict[idx][0]
			pmax_dict[idx] = pmax

		return pmax_dict


	def _scale_data(self, dispatch_dict, input_data_dict):
		
		'''
		scale the data by pmax to get capacity factors

		Arguments:

			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		Returns:
			scaled_dispatch_dict: {run_index: [scaled dispatch data]}
		'''

		index_list = list(dispatch_dict.keys())

		pmax_dict = self._read_pmax(dispatch_dict, input_data_dict)

		scaled_dispatch_dict = {}

		for idx in index_list:
			dispatch_year_data = dispatch_dict[idx]
			pmax_year = pmax_dict[idx]

			scaled_dispatch_year_data = dispatch_year_data/pmax_year
			scaled_dispatch_dict[idx] = scaled_dispatch_year_data

		return scaled_dispatch_dict	


	def _transform_data(self, dispatch_dict, input_data_dict):
		
		'''
		Transform the data to clustering package required form.

		Arguments:

			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		Returns:
			train_data: training data for clustering
		'''

		scaled_dispatch_dict = self._scale_data(dispatch_dict, input_data_dict)

		index_list = list(scaled_dispatch_dict.keys())

		if self.filter_opt == True:
			full_day = 0
			zero_day = 0
			day_dataset = []
			for idx in index_list:
				sim_year_data = scaled_dispatch_dict[idx]
				day_num = int(len(sim_year_data)/self._time_length)
				
				for day in range(day_num):
					sim_day_data = sim_year_data[day*24:(day+1)*24]
					
					if sum(sim_day_data) == 0:
						zero_day += 1
					elif sum(sim_day_data) == 24:
						full_day += 1
					else:
						day_dataset.append(sim_day_data)

			train_data = to_time_series_dataset(day_dataset)

			return train_data

		elif self.filter_opt == False:
			day_dataset = []

			for idx in index_list:
				sim_year_data = scaled_dispatch_dict[idx]
				day_num = int(len(sim_year_data)/self._time_length)

				for day in range(day_num):
					sim_day_data = sim_year_data[day*24:(day+1)*24]
					day_dataset.append(sim_day_data)

			train_data = to_time_series_dataset(day_dataset)

			return train_data


	def clustering_data(self, dispatch_dict, input_data_dict):

		'''
		Time series clustering for the dispatch data

		Arguments:

			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		Returns:
			clustering_model: trained clustering model
		'''

		train_data = self._transform_data(dispatch_dict, input_data_dict)

		clustering_model = TimeSeriesKMeans(n_clusters = self.num_clusters, metric = self.metric, random_state = 0)
		labels = clustering_model.fit_predict(train_data)

		return clustering_model


	def save_clustering_model(self, clustering_model, fpath = None):

		'''
		Save the model in .json file. fpath can be specified by the user. 

		Arguments:

			clustering_model: trained model from self.clustering_data()

		Return:
			result_path: result path for the json file. 
		'''

		if fpath == None:
			result_path =  f'Time_series_clustering\\clustering_results\\auto_result_{self.num_sims}years_shuffled_0_{self.num_clusters}clusters_OD.json'
			clustering_model.to_json(result_path)

		else:
			result_path = fpath
			clustering_model.to_json(result_path)

		return result_path


	def get_cluster_centers(self, result_path):

		'''
		Get the cluster centers.

		Arguments:

			result_path: the path of clustering model

		Returns:
			centers_list: {cluster_center:[results]} 
		'''

		with open(result_path, 'r') as f:
			cluster_results = json.load(f)
		
		centers = np.array(cluster_results['model_params']['cluster_centers_'])

		centers_dict = {}
		for i in range(len(centers)):
			centers_dict[i] = centers[i]

		return centers_dict

	
	# In progress
	# def plot_results(self, result_path):

	# 	centers_dict = self.get_cluster_centers(result_path)

	# 	time_length = range(24)

	# 	f,ax1 = plt.subplots(figsize = ((16,6)))
	# 	for j in range(self.num_clusters):
	# 		ax1.plot(time_len,new_center_dict[num][j], '-')

	# 	ax1.set_ylabel('Dispatched Power(MW)')
	# 	ax1.set_xlabel('Time(h)')
	# 	plt.show()

	# 	return

class TrainNNSurrogates:
	
	'''
	Train neural network surrogates for the dispatch frequency
	'''

	def __init__(self, num_clusters, filter_opt = True):

		self.filter_opt = filter_opt
		self.num_clusters = num_clusters
		self._time_length = 24

		
	def _read_clustering_model(self, clustering_model_path):

		'''
		Read the time series clustering model from the given path

		Returns:

			Clustering model
		'''
		clustering_model = TimeSeriesKMeans.from_json(clustering_model_path)

		return clustering_model


	def _read_pmax(self, dispatch_dict, input_data_dict):

		'''
		Read pmax from input_dict
		
		Arguments:
			
			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		
		Returns:
			pmax_dict: {run_index: pmax}
		'''

		index_list = list(dispatch_dict.keys())

		pmax_dict = {}

		for idx in index_list:
			pmax = input_data_dict[idx][0]
			pmax_dict[idx] = pmax

		return pmax_dict


	def _scale_data(self, dispatch_dict, input_data_dict):

		'''
		scale the data by pmax to get capacity factors

		Arguments:

			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		Returns:
			scaled_dispatch_dict: {run_index: [scaled dispatch data]}
		'''

		index_list = list(dispatch_dict.keys())

		pmax_dict = self._read_pmax(dispatch_dict, input_data_dict)

		scaled_dispatch_dict = {}

		for idx in index_list:
			dispatch_year_data = dispatch_dict[idx]
			pmax_year = pmax_dict[idx]

			scaled_dispatch_year_data = dispatch_year_data/pmax_year
			scaled_dispatch_dict[idx] = scaled_dispatch_year_data

		return scaled_dispatch_dict


	def _generate_feature_data(self, clustering_model_path, dispatch_dict, input_data_dict):

		'''
		Calculate the labels for NN training. 

		Arguments:
			clustering_model_path: saved clustering model, json file.

			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep

		Return:
			
			dispatch_frequency_dict: {run_index: [dispatch frequency]}

		'''
		clustering_model = self._read_clustering_model(clustering_model_path)
		scaled_dispatch_dict = self._scale_data(dispatch_dict, input_data_dict)
		sim_index = list(dispatch_dict.keys())
		single_day_dataset = {}
		dispatch_frequency_dict = {}
		
		if self.filter_opt == True:
			for idx in sim_index:
				sim_year_data = scaled_dispatch_dict[idx]
				single_day_dataset[idx] = []
				day_num = int(len(sim_year_data)/self._time_length)
				day_0 = 0
				day_1 = 0
				for day in range(day_num):
					sim_day_data = sim_year_data[day*self._time_length:(day+1)*self._time_length]
					if sim_day_data.sum() == 0:
						day_0 += 1
					elif sim_day_data.sum() == 24:
						day_1 += 1
					else:
						single_day_dataset[idx].append(sim_day_data)
			
				# frequency of 0/1 days
				ws0 = day_0/day_num
				ws1 = day_1/day_num


				if len(single_day_dataset[idx]) == 0:
					labels = np.array([])

				else:
					to_pred_data = to_time_series_dataset(single_day_dataset[idx])
					labels = clustering_model.predict(to_pred_data)

				elements, count = np.unique(labels,return_counts=True)

				pred_result_dict = dict(zip(elements, count))
				count_dict = {}
				for j in range(self.num_clusters):
					if j in pred_result_dict.keys():
						count_dict[j] = pred_result_dict[j]/day_num
					else:
						count_dict[j] = 0

				# the first element in w is frequency of 0 cf days
				dispatch_frequency_dict[idx] = [ws0]

				for key, value in count_dict.items():
					dispatch_frequency_dict[idx].append(value)

				# the last element in w is frequency of 1 cf days
				dispatch_frequency_dict[idx].append(ws1)

			return(dispatch_frequency_dict)

		else:
			for idx in sim_index:
				sim_year_data = scaled_dispatch_dict[idx]
				single_day_dataset[idx] = []
				day_num = int(len(sim_year_data)/self._time_length)
				for day in range(day_num):
					single_day_dataset[idx].append(sim_day_data)

				to_pred_data = to_time_series_dataset(single_day_dataset[idx])
				labels = clustering_model.predict(to_pred_data)

				elements, count = np.unique(labels,return_counts=True)
				pred_result_dict = dict(zip(elements, count))
				count_dict = {}
				
				for j in range(self.num_clusters):
					if j in pred_result_dict.keys():
						count_dict[j] = pred_result_dict[j]/day_num
					else:
						count_dict[j] = 0

				for key, value in count_dict.items():
					dispatch_frequency_dict[idx].append(value)

			return dispatch_frequency_dict


	def _transform_dict_to_array(self, clustering_model_path, dispatch_dict, input_data_dict):

		'''
		transform the dictionary data to array that keras can train

		Arguments:

			dispatch_dict: dictionary stores dispatch data.

			input_dict: dictionary stores input data for parameter sweep
		
		Returns:

			x: features (input)
			y: labels (dispatch frequency)
		'''

		dispatch_frequency_dict = self._generate_feature_data(clustering_model_path, dispatch_dict, input_data_dict)
		
		index_list = list(dispatch_dict.keys())

		x = []
		y = []

		for idx in index_list:
			x.append(input_data_dict[idx])
			y.append(dispatch_frequency_dict[idx])

		return np.array(x), np.array(y)


	def train_NN(self, clustering_model_path, dispatch_dict, input_data_dict):

		'''
		train the dispatch frequency NN surrogate model.
		print the R2 results of each cluster.

		Arguments:
			clustering_model: time series clustering model from tsc
			dispatch_data: path of the training dispatch csv file (need to calculate 0/1 days in each simulation)
			x: parameter sweep input for each simulation, from function 'read_from_csv'

		return:
			None
		'''
		x, ws = self._transform_dict_to_array(clustering_model_path, dispatch_dict, input_data_dict)

		x_train, x_test, ws_train, ws_test = train_test_split(x, ws, test_size=0.2, random_state=42)

		# scale the data both x and ws
		xm = np.mean(x_train,axis = 0)
		xstd = np.std(x_train,axis = 0)
		wsm = np.mean(ws_train,axis = 0)
		wsstd = np.std(ws_train,axis = 0)
		x_train_scaled = (x_train - xm) / xstd
		ws_train_scaled = (ws_train - wsm)/ wsstd

		# train a keras MLP (multi-layer perceptron) Regressor model
		model = keras.Sequential(name='dispatch_frequency')
		model.add(layers.Input(8))
		model.add(layers.Dense(300, activation='sigmoid'))
		model.add(layers.Dense(300, activation='sigmoid'))
		model.add(layers.Dense(300, activation='sigmoid'))
		model.add(layers.Dense(32))
		model.compile(optimizer=Adam(), loss='mse')
		history = model.fit(x=x_train_scaled, y=ws_train_scaled, verbose=1, epochs=500)

		print("Making NN Predictions...") 

		# normalize the data
		### need to check how to normalize the test data
		x_test_scaled = (x_test - xm) / xstd
		ws_test_scaled = (ws_test - wsm) / wsstd

		print("Evaluate on test data")
		evaluate_res = model.evaluate(x_test_scaled, ws_test_scaled)
		print(evaluate_res)
		predict_ws = np.array(model.predict(x_test_scaled))
		predict_ws_unscaled = predict_ws*wsstd + wsm

		test_R2 = []
		for rd in range(0,32):
			# compute R2 metric
			wspredict = predict_ws_unscaled.transpose()[rd]
			SS_tot = np.sum(np.square(ws_test.transpose()[rd] - wsm[rd]))
			SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
			residual = 1 - SS_res/SS_tot
			test_R2.append(residual)

		accuracy_dict = {"R2":test_R2}

		print(test_R2)

		return model

	def save_model(self, model):

		this_file_path = os.getcwd()
		model_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled\\keras_dispatch_frequency_sigmoid')
		model.save(model_save_path)

		xmin = list(np.min(x_train_scaled, axis=0))
		xmax = list(np.max(x_train_scaled, axis=0))

		data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
			"ws_mean":list(wsm),"ws_std":list(wsstd)}

		param_save_path = os.path.join(this_file_path, 'NN_model_params_keras_scaled\\keras_training_parameters_ws_scaled.json')
		with open(paran_save_path, 'w') as f2:
			json.dump(data, f2)


	# In progress 
	# def plot_R2_results(self):

		# this_file_path = 
		# input_file = 
		# dispatch_csv = 
		# mdclustering = 
		



def main():

	current_path = os.getcwd()
	dispatch_data = os.path.join(current_path, 'Time_series_clustering\\datasets\\Dispatch_shuffled_data_0.csv')
	input_data = os.path.join(current_path, 'Time_series_clustering\\datasets\\prescient_generator_inputs.h5')
	num_clusters = 30

	# test TimeSeriesClustering
	clusteringtrainer = TimeSeriesClustering(6400,num_clusters)
	dispatch_dict, input_dict = clusteringtrainer.read_data_to_dict(dispatch_data, input_data)
	clustering_model = clusteringtrainer.clustering_data(dispatch_dict, input_dict)
	result_path = clusteringtrainer.save_clustering_model(clustering_model)
	# centers_dict = clusteringtrainer.get_cluster_centers(result_path)


	# test class TrainNNSurrogates
	NNtrainer = TrainNNSurrogates(num_clusters)
	model = NNtrainer.train_NN(model_path, dispatch_dict, input_dict)
	NNtrainer.save_model(model)




if __name__ == "__main__":
	main()

	


