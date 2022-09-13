Create time series clustering + neural network surrogate for dispatch frequency ws.

1. conceptual_design_dynamic: contains the codes for dynamic conceptual design problem with surrogates. 
    Current clustering result is from Dispatch_shuffled_data_0.csv, (6400 years). 32 clusters (including 0/1 capacity factor clusters).
    The clustering results is stored in result_6400years_shuffled_0_30clusters_OD.json. 
    
    We have already solved conceptual design problem (wind+battery) with revenue, nstartup, dispatch frequency surrogate models.
    Try 

2. figures:
    A folder which has R^2 plots of dispatch frequency surrogates.


3. NN_model_params_keras_scaled:
    This folder has the NN surrogate model parameters that are trained using keras. 
    In this folder, all xmin and xmax in keras_training_parameters_surrogate_scaled.json are scaled. 
    We do this to satisfy the requirements of OMLT v1.0.

6. Time_series_clustering
    This folder has the codes for time series clustering. 

    6.1 clustering_result
          store the clustering results.
    
    6.2 datasets
          store the datasets for training.

    6.3 figures
          have the NN model prediction plots with R2

    6.4 only_dispatch
          have the time series clustering codes for dispatches profiles
	filter_01_6400_years.py: do filter out days with zero/ full capacify factor days before clustering
	tslearn_test_6400_years.py: do not filter out days with zero/ full capacify factor days before clustering (not preferred)

    6.5 train_kerasNN
          have the codes for training Kears NN model using clustering results
	TSA_NN_surrogate_keras.py: train keras NN model. Can read R2 from the output.]
	plot_ws_results_keras.py: split data into train and test, then plot results.

    6.6 wind+pv
          do time series clustering with (dispatch, wind, pv) data. (3 dimensions) 


Note: all notebook files are not well developed. Not easy to read them. I will try to improve them later. 