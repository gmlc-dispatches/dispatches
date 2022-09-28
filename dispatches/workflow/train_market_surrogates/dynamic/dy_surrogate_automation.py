# import packages

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
# temporarily put them here

class ReadData:
	def __init__(self, num_sims = 1):

		self.num_sims = num_sims


	def _read_data_to_array(self, dispatch_data_file, input_data_file):

		'''
		Read the dispatch data from the csv file

		Arguments:
			csv_file: the file stores dispatch profiles by simulation years

			num_years: specify the number of sims that user wants to read. Default = 10

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



# class TimeSeriesClustering(TSA64K):
# 	def __init__(self, dispatch_data, metric, years, num_clusters, filter_opt):
# 		super(TimeSeriesClustering, self).__init__(dispatch_data, metric, years, num_clusters, filter_opt)

# 		__this_file_path__ = os.getcwd()
		
# 		result_path = os.path.join(path0, f'..\\clustering_results\\result_{self.years}years_{self.num_clusters}clusters_OD.json')

# 		self._default_path = result_path

# 	def save_results(self, km, path = self._default_path):

# 		'''
# 		Save the model.

# 		Return:
# 			None
# 		'''
# 		km.to_json(result_path)


	
# 	def plot_results(self, result_path):

# 		with open(result_path, 'r') as f:
# 			cluster_results = json.load(f)

# 		centers = np.array(cluster_results['model_params']['cluster_centers_'])

# 		time_length = range(24)

# 		f,ax1 = plt.subplots(figsize = ((16,6)))
# 		for j in range(self.num_clusters):
# 			ax1.plot(time_len,new_center_dict[num][j], '-')

# 		ax1.set_ylabel('Dispatched Power(MW)')
# 		ax1.set_xlabel('Time(h)')
# 		plt.show()

# 		return


# 	def get_cluster_centers(self, result_path):

# 		with open(result_path, 'r') as f:
# 			cluster_results = json.load(f)
		
# 		centers = np.array(cluster_results['model_params']['cluster_centers_'])

# 		return centers



class TrainNNSurrogates:
	"""docstring for TrainNNSurrogates"""
	def __init__(self, dispatch_dict, input_data_dict, filter_opt, num_clusters):
		
		self.dispatch_dict = dispatch_dict
		self.input_data_dict = input_data_dict
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


	def _read_pmax(self):

		index_list = list(self.dispatch_dict.keys())

		pmax_dict = {}

		for idx in index_list:
			pmax = self.input_data_dict[idx][0]
			pmax_dict[idx] = pmax

		return pmax_dict


	def _scale_data(self):
		index_list = list(self.dispatch_dict.keys())

		pmax_dict = self._read_pmax()

		scaled_dispatch_dict = {}

		for idx in index_list:
			dispatch_year_data = self.dispatch_dict[idx]
			pmax_year = pmax_dict[idx]

			scaled_dispatch_year_data = dispatch_year_data/pmax_year
			scaled_dispatch_dict[idx] = scaled_dispatch_year_data

		return scaled_dispatch_dict


	# def _transform_data(self):

	# 	dataset = []

	# 	scaled_dispatch_dict = self._scale_data()

	# 	for idx in list(scaled_dispatch_dict.keys()):
	# 		simulation_year_data = scaled_dispatch_dict[idx]
	# 		num_days = int(len(simulation_year_data)/self._time_length)
	# 		for b in range(num_days):
	# 			single_day_data = simulation_year_data[i*24:(i+1)*24]
	# 			dataset.append(single_day_data)

	# 	clustering_pred_data = to_time_series_dataset(dataset)

	# 	return clustering_pred_data



	def _generate_feature_data(self, clustering_model_path):

		'''
		calculate the demand frequency ws for the given data

		Arguments:
			clustering_model: saved clustering model, json file

			sims: number of simulations

		Return:
			
			ws: with shape of (years, number of clusters)

		'''
		if self.filter_opt == True:

			clustering_model = self._read_clustering_model(clustering_model_path)

			scaled_dispatch_dict = self._scale_data()

			pred_res = []

			day_01_count = []

			sim_index = list(self.dispatch_dict.keys())

			single_day_dataset = {}

			dispatch_frequency_dict = {}

			for idx in sim_index:
				sim_data = scaled_dispatch_dict[idx]
				single_day_dataset[idx] = []
				day_num = int(len(sim_data)/self._time_length)
				day_0 = 0
				day_1 = 0
				for i in range(day_num):
					day_data = sim_data[i*self._time_length:(i+1)*self._time_length]
					if day_data.sum() == 0:
						day_0 += 1
					elif day_data.sum() == 24:
						day_1 += 1
					else:
						single_day_dataset[idx].append(day_data)
			
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

			return('placeholder')


	def _transform_dict_to_array(self, clustering_model_path):

		dispatch_frequency_dict = self._generate_feature_data(clustering_model_path)
		
		index_list = list(self.dispatch_dict.keys())

		x = []
		y = []

		for idx in index_list:
			x.append(self.input_data_dict[idx])
			y.append(dispatch_frequency_dict[idx])

		return np.array(x), np.array(y)


	def train_ws(self, clustering_model_path):

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
		x, ws = self._transform_dict_to_array(clustering_model_path)

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
		model_save_path = os.path.join(this_file_path, '..\\..\\NN_model_params_keras_scaled\\keras_dispatch_frequency_sigmoid')
		model.save(model_save_path)

		xmin = list(np.min(x_train_scaled, axis=0))
		xmax = list(np.max(x_train_scaled, axis=0))
		data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
			"ws_mean":list(wsm),"ws_std":list(wsstd)}
		param_save_path = os.path.join(this_file_path, '..\\..\\NN_model_params_keras_scaled\\keras_training_parameters_ws_scaled.json')
		with open(paran_save_path, 'w') as f2:
			json.dump(data, f2)

		

# def conceptual_design(plant_type = 'RE'):

# 	'''
# 	Build and solve conceptual design problems. Put results in a dictionary and save results in json file.

# 	Argument: 
# 		plant_type: 'RE, NU, FOSSIL'
# 			now only have RE mode. 
	
# 	return
# 		result_dict: dictionary that has design results. 

# 	'''

# 	RE_default_input_params = {
#     "wind_mw": 440.5,
#     "wind_mw_ub": 10000,
#     "batt_mw": 40.05,
#     "pem_mw": None,
#     "pem_bar": None,
#     "pem_temp": None,
#     "tank_size": None,
#     "tank_type": None,
#     "turb_mw": None,

#     "wind_resource": None,
#     "h2_price_per_kg": None,
#     "DA_LMPs": None,

#     "design_opt": True,
#     "extant_wind": False
# 	} 

# 	start_time = time.time()
# 	model = conceptual_design_dynamic_RE(RE_default_input_params, num_rep_days = 32, verbose = False, plant_type = 'RE')

# 	nlp_solver = SolverFactory('ipopt')
# 	# nlp_solver.options['max_iter'] = 500
# 	nlp_solver.options['acceptable_tol'] = 1e-8
# 	nlp_solver.solve(model, tee=True)
# 	end_time = time.time()

# 	print('------------------------------------------------------------------------')
# 	print('Time for solving the model is {} seconds'.format(end_time - start_time))
# 	print('------------------------------------------------------------------------')

# 	record_result(model,32)

# 	# save the model
# 	to_json(model, fname = 'run3.json.gz', human_read = True)

# 	# result_dict = {wind_pmax: a, battery_pmax:b, battery_energy_capacity: c}
# 	# save results as a json file
# 	# result_dict.josn

# 	return result_dict

# def double_loop_verify(result_dict):
# 	'''
# 	step 5:
# 	double_loop verify

# 	Arguments:
# 		result_dict: design results from function 'conceptual_design'

	
# 	'''
# 	return



def main():

	current_path = os.getcwd()
	dispatch_data = os.path.join(current_path, 'Time_series_clustering\\datasets\\Dispatch_shuffled_data_0.csv')
	input_data = os.path.join(current_path, 'Time_series_clustering\\datasets\\prescient_generator_inputs.h5')
	num_clusters = 30

	data_reader = ReadData(6400)

	dispatch_dict, input_dict = data_reader.read_data_to_dict(dispatch_data, input_data)

	NNtrainer = TrainNNSurrogates(dispatch_dict, input_dict, True, 30)

	model_path = os.path.join(current_path, 'Time_series_clustering\\clustering_results\\result_6400years_shuffled_0_30clusters_OD.json')

	NNtrainer.train_ws(model_path)
	# debug
	# c = 0
	# for key, value in dispatch_frequency_dict.items():
	# 	print(key,value)
	# 	# print(dispatch_dict[key])
	# 	c += 1
	# 	if c > 3:
	# 		break

if __name__ == "__main__":
	main()

	


