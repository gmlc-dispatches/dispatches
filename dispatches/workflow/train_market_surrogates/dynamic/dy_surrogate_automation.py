'''

# build dynamic_surrogate workflow:

0. extract data from the orgin simulation files (should discuss with Alex if we need this step)

1. Time series clustering

	1.1 read data from csv files 

	1.2 filter out zero/full capacity days

	1.3 transform data to structure that the clustering package requires.

 	1.4 do clustering

 	1.5 save results. (and visualize results, boxplots...)


2. Train Keras NN surrogate models.

	2.1 load clustering model

	2.2 calculate ws for each simulation year (including 0/1 capacity days). 

	2.3 read input x for each simulation year

	2.4 train NN surrogate model

	2.5 save model, parameters, visualize R2 results. 


3. Train revenue and #start-up surrogate models

	3.1 read data from csv files
	
	3.2 build correct data structure for training

	3.3 read input x for each simulation year

	3.4 train NN surrogate model

	3.5 save model, parameters, visualize R2 results. 


4. Solve conceptual design problem

	4.1 load surrogate models.

	4.2 build and solve conceptual design problem
		for RE case, we need wind pmax, battery pmax, battery energy capacity, n scenarios, participation_mode, reserve_factor. 

	4.3 save design results in csv files.


5. Double loop validation

	5.1 read design results from step 4.3

	5.2 solve double loop simulation. 
'''


# import packages

import os

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from Time_series_clustering.only_dispatch.filter_01_6400_years import TSA64K
from Time_series_clustering.train_kerasNN.TSA_NN_surrogate_keras import load_cluster_model, calculate_ws, read_input_x, train_NN_surrogate
from conceptual_design_dynamic.new_full_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE, record_result
from pyomo.environ import value, SolverFactory
from idaes.core.util import to_json, from_json
import time
import pandas as pd
import numpy as np
import json


# need to find out a place to store the data instead of just put them in the dispatches repo
# temporarily put them here


class TimeSeriesClustering(TSA64K):
	def __init__(self, dispatch_data, metric, years, num_clusters, filter_opt):
		super(TimeSeriesClustering, self).__init__(dispatch_data, metric, years, num_clusters, filter_opt)
	
	def plot_results(self, result_path):

		with open(result_path, 'r') as f:
		    cluster_results = json.load(f)
		
		centers = np.array(cluster_results['model_params']['cluster_centers_'])
		    
		time_len = range(24)

		f,ax1 = plt.subplots(figsize = ((16,6)))
	    for j in range(self.num_clusters):
	        ax1.plot(time_len,new_center_dict[num][j], '-')

	    ax1.set_ylabel('Dispatched Power(MW)')
	    ax1.set_xlabel('Time(h)')
	    plt.show()

	    return

# def read_from_csv(dispatch_data, input_data, num_sims = 6400):
# 	'''
# 	read csv files. Output correct data structure for clusterin and NN training.
# 	dispatch_data: Dispatch_shuffled_data0.csv
# 	input_data: rescient_generator_inputs.h5

# 	Arguments:
# 		dispatch_data: path of the training dispatch csv file
# 		input_data: path of the input data of parameter sweep
# 		num_sims: how many simulations do you want to put in the training. Default 6400

# 	return:
# 		training_data: data for time series clustering
# 		input_data: input data for training NN
# 	'''
# 	tsa_task = TSA64K(dispatch_data, metric, num_sims)
# 	dispatch_array = tsa_task.read_data()
# 	train_data,day_01 = tsa_task.transform_data(dispatch_array)
# 	x = read_input_x(input_data, dispatch_data)

# 	return train_data, x



def tsc(train_data, num_clusters, num_sims = 6400):
	'''
	step 1. Time series clustering
	Time series clustering over given data.
	get certain number of representative days.

	Arguments:
		train_data: train_data from function 'read_from_csv'
		num_clusters: number of representative days
		num_sims: how many simulations do you want to put in the training. Default 6400

	return:
		km: clustering model.
	'''

	metric = 'euclidean'

	# at this moment use Dispatch_data_shuffled_0.csv
	data_num = 0

	clustering_model = TimeSeriesKMeans(n_clusters = num_clusters, metric = metric, random_state = 0)
	labels = clustering_model.fit_predict(train_data)

	# saved clustering model is in Time_series_clustering/cluster_results.
	path0 = os.getcwd()
	result_path = os.path.join(path0, f'..\\clustering_results\\result_{self.years}years_{data_num}_{clusters}clusters_OD.json')
	
	# save the clustering model
	clustering_model.to_json(result_path)

	return clustering_model


def train_ws(clustering_model, dispatch_data, x):
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
	
	# calculate ws of the training data
	ws = calculate_ws(clustering_model, dispatch_data, years = 6400)

	# save_index = True means we have saved the model
	R2 = train_NN_surrogate(x, ws, save_index = True)

	# visualize the results


	print('---------------------------------------------------------')
	for i,r2 in enumerate(R2):
		print(f'R2 of cluster_{i} is {r2}')
	print('---------------------------------------------------------')

	return


# def train_rev():
# 	return


# def train_nstartup():
# 	return


def conceptual_design(plant_type = 'RE'):

	'''
	Build and solve conceptual design problems. Put results in a dictionary and save results in json file.

	Argument: 
		plant_type: 'RE, NU, FOSSIL'
			now only have RE mode. 
	
	return
		result_dict: dictionary that has design results. 

	'''

	RE_default_input_params = {
    "wind_mw": 440.5,
    "wind_mw_ub": 10000,
    "batt_mw": 40.05,
    "pem_mw": None,
    "pem_bar": None,
    "pem_temp": None,
    "tank_size": None,
    "tank_type": None,
    "turb_mw": None,

    "wind_resource": None,
    "h2_price_per_kg": None,
    "DA_LMPs": None,

    "design_opt": True,
    "extant_wind": False
	} 

	start_time = time.time()
	model = conceptual_design_dynamic_RE(RE_default_input_params, num_rep_days = 32, verbose = False, plant_type = 'RE')

	nlp_solver = SolverFactory('ipopt')
	# nlp_solver.options['max_iter'] = 500
	nlp_solver.options['acceptable_tol'] = 1e-8
	nlp_solver.solve(model, tee=True)
	end_time = time.time()

	print('------------------------------------------------------------------------')
	print('Time for solving the model is {} seconds'.format(end_time - start_time))
	print('------------------------------------------------------------------------')

	record_result(model,32)

	# save the model
	to_json(model, fname = 'run3.json.gz', human_read = True)

	# result_dict = {wind_pmax: a, battery_pmax:b, battery_energy_capacity: c}
	# save results as a json file
	# result_dict.josn

	return result_dict

def double_loop_verify(result_dict):
	'''
	step 5:
	double_loop verify

	Arguments:
		result_dict: design results from function 'conceptual_design'

	
	'''
	return



def main():
	dispatch_data = os.path.join(__this_file_dir__, '\\Time_series_clustering\\datasets\\Dispatch_shuffled_data_0.csv')
	input_data = os.path.join(this_file_path, '\\Time_series_clustering\\datasets\\prescient_generator_inputs.h5')
	num_clusters = 15

	# read the csv data, default we are working on 6400 simulations
	train_data, x = read_from_csv(dispatch_data, input_data):
	clustering_model = tsc(num_clusters)

	# train dispatch frequency NN surrogate
	train_ws(clustering_model, dispatch_data, x)


	# solve conceptual design optimization
	result_dict = conceptual_design()

	# double loop verification
	double_loop_verify(result_dict)


	


