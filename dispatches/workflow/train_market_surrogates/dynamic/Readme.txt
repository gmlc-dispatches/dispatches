Create time series clustering + neural network surrogate for dispatch frequency ws.

1. conceptual_design_dynamic: contains the codes for dynamic conceptual design problem with surrogates. 
    Current clustering result is from Dispatch_shuffled_data_0.csv, (6400 years). 32 clusters (including 0/1 capacity factor clusters).
    The clustering results is stored in result_6400years_shuffled_30clusters_OD.json. 
    We are testing the codes following the below steps
	(a) Solve the multi-period problem with N=3 representative days with fixed frequency and no startup. (Done)
	(b) Add Startup surrogate debug syntax (Done)
	(c) Solve the multi-period problem with revenue and startup surrogates, N=3 representative days, fixed weights. (Done)
	(d) Solve N=20 to N=50 representative days solve the model (In working)
	(e) Add dispatch frequency surrogate and debug syntax. 
	(f) Solve with N=3 then N=32 days (full size problem).
    Next steps will be use double loop to verify the design results. 

2. figures:
    A folder which has R^2 plots of dispatch frequency surrogates.

3. NN_model_params_keras:
    This folder has the NN surrogate model parameters that are trained using keras.

4. NN_model_params_keras_scaled:
    This folder has the NN surrogate model parameters that are trained using keras. 
    Differ from the above one, in this folder, all xmin and xmax in keras_training_parameters_surrogate_scaled.json are scaled. 
    We do this to satisfy the requirements of OMLT v1.0.

5. NN_model_params_scikit:
    This folder has the NN surrogate model parameters that are trained using scikit-learn.
    In OMLT v0.3.1 use can use scikit-learn trained NN. OMLT v0.3.1 is required for IDAESv2.0 dev.
    But I turned to OMLT v1.0, not confliction and bugs to IDAESv2.0 dev at this moment.

6. Time_series_clustering
    This folder has the codes for time series clustering. 

