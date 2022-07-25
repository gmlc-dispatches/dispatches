import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import TimeSeriesKMeans
import os
from tslearn_test_6400_years import TSA64K
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam



# use the clustering result of 'Dispatch_shuffled_data_0.csv' (6400 years, 30+2 clusters)

# input layer 8 nodes, output 1 nodes for frequency of each representative day.

def load_cluster_model(model_file):
	
	'''
	load the clustering model

	Arguments:
		model_file: the json file that stores the clustering model 
		### future work: complete codes that can also read pkl file
	
	Return:
		clustering_model
	'''

	# model_file should be in the same folder
	result_path = model_file
	clustering_model = TimeSeriesKMeans.from_json(result_path)

	return clustering_model



def calculate_ws(clustering_model, pred_csv, years = 6400):

	'''
	calculate the demand frequency ws for the given data

	Arguments:
		pred_csv: csv file that stores the data we are going to predict

		years: int, how many simulation years are we going to predict, default 6400

	Return:
		ws: with shape of (years, number of clusters)

	'''

	# Use functions in TSA64K to get train_data
	metric = 'euclidean'
	tsa_task = TSA64K(pred_csv, metric, years)
	dispatch_array = tsa_task.read_data()
	tsa_task.read_input_pmax()
	train_data, data_index = tsa_task.transform_data(dispatch_array)

	pred_res = []
	day_01_count = []

	# pred_res: list of (years, 364), are labels predicted by the clustering_model
	for i in range(years):
	    year_data = train_data[i*364:(i+1)*364]
	    day_0 = 0
	    day_1 = 0
	    del_index = []
	    for idx, day in enumerate(year_data):
	    	if day.sum() == 0:
	    		day_0 += 1
	    		del_index.append(idx)
	    	elif day.sum() == 24:
	    		day_1 += 1
	    		del_index.append(idx)
	    	else:
	    		continue

	    # count how many 0/1 capacity factor days in one year.
	    day_01_count.append(np.array([day_0,day_1]))
	    
	    # just pred the days that are not 0/1.
	    new_year_data = np.delete(year_data, del_index, axis = 0)

	    # In some cases, the whole year is 0 capacity, add an empty array
	    if len(new_year_data) == 0:
	    	pred_res.append(np.array([]))
	    else:
	    	pred_res.append(clustering_model.predict(new_year_data))

	day_01_count = np.array(day_01_count)

	# record the number of clusters
	num_clusters = clustering_model.n_clusters

	# count the 0/1 capacity days
	ws = []
	for c, yr in enumerate(pred_res):
	    elements, count = np.unique(yr,return_counts=True)
	    res_dict = dict(zip(elements,count))
	    count_dict = {}
	    for j in range(num_clusters):
	        if j in res_dict.keys():
	            count_dict[j] = res_dict[j]/364
	        else:
	            count_dict[j] = 0

	    # the first element in w is frequency of 0 cf days
	    w = [day_01_count[c][0]/364]
	    for k in count_dict.items():
	        w.append(k[1])

	    # the last element in w is frequency of 0 cf days
	    w.append(day_01_count[c][1]/364)
	    ws.append(w)

	# ws now np.array with size of (years,32)
	ws = np.array(ws)

	return ws


def read_input_x(input_data_x, pred_csv, years = 6400):

	'''
	read the input x from the csv

	Arguments:
		input_data_x: csv file that stores the input data 

		pred_csv: csv file that stores the data we are going to predict

		years: int, how many simulation years are we going to use, default 6400

	Return:
		x: with shape of (years, 8)

	'''
	metric = 'euclidean'
	tsa_task = TSA64K(pred_csv, metric, years)
	dispatch_array = tsa_task.read_data()
	tsa_task.read_input_pmax()
	train_data, data_index = tsa_task.transform_data(dispatch_array)

	df_input_data = pd.read_hdf(input_data_x)

	# rows: data_index
	# cols: from jordan's code
	x = df_input_data.iloc[data_index,[1,2,3,4,5,6,7,9]].to_numpy()

	return x

def train_NN_surrogate(x, ws, save_index = False):
	# '''
	# train the neural network surrogate and predict on test data
	
	# Arguments:
	# 	x: input data
	
	# 	ws: output data, dispatch frequency
	
	# 	save_index: bool, default = True. save the NN model params

	# Return:
	# 	R2: R2 metric
	# 	scores: cross validation scores

	# '''
	# # split the train/test sets
    
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
    history = model.fit(x=x_train_scaled, y=ws_train_scaled, verbose=0, epochs=500)

    print("Making NN Predictions...") 

    # normalize the data
    ### need to check how to normalize the test data
    x_test_scaled = (x_test - xm) / xstd
    ws_test_scaled = (ws_test - wsm) / wsstd
    
    print("Evaluate on test data")
    evaluate_res = model.evaluate(x_test_scaled, ws_test_scaled)
    print(evaluate_res)
    predict_ws = model.predict(x_test_scaled)
    predict_ws_unscaled = predict_ws*wsstd + wsm
    print(np.shape(predict_ws_unscaled))

    test_R2 = []
    for rd in range(0,32):
        # compute R2 metric
        wspredict = predict_ws_unscaled.transpose()[rd]
        SS_tot = np.sum(np.square(wspredict - wsm[rd]))
        SS_res = np.sum(np.square(ws_test.transpose()[rd] - wspredict))
        residual = 1 - SS_res/SS_tot
        test_R2.append(residual)

    accuracy_dict = {"R2":test_R2}

    if save_index == True:
    	#save the NN model 
    	model.save('NN_model_params/keras_sigmoid_dispatch_frequency')

  #   	# save the arruracy params
  #   	with open('NN_model_params/keras_NN_ws_accuracy.json','w') as f1:
  #   		json.dump(accuracy_dict, f1)

  #   	# save training bounds and scaling.
    	xmin = list(np.min(x_train_scaled, axis=0))
    	xmax = list(np.max(x_train_scaled, axis=0))
    	data = {"xm_inputs":list(xm),"xstd_inputs":list(xstd),"xmin":xmin,"xmax":xmax,
		"ws_mean":list(wsm),"ws_std":list(wsstd)}

    	with open('NN_model_params/keras_training_params_ws_sigmoid.json', 'w') as f2:
    		json.dump(data, f2)
    	
    
    return test_R2


def main():
	inp_file = 'prescient_generator_inputs.h5'
	pred_csv = 'Dispatch_shuffled_data_0.csv'
	x = read_input_x(inp_file, pred_csv)
	# print('---------------------------')
	# print('first 5 input vector')
	# print(x[:5])

	clustering_model = load_cluster_model('result_6400years_shuffled_30clusters_OD.json')

	ws = calculate_ws(clustering_model, pred_csv)

	test_R2 = train_NN_surrogate(x, ws,save_index = True)

	print(test_R2)
	# print(scores)


if __name__ == '__main__':
	main()
