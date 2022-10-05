import pandas as pd
import numpy as np
import os

file_name_1 = 'input_data\\sweep_parameters_results_nuclear_sweep_10_500'
file_name_2 = 'input_data\\sweep_parameters_results_nuclear_sweep_10_1000'
file_name_3 = 'input_data\\sweep_parameters_results_nuclear_sweep_15_500'
file_name_4 = 'input_data\\sweep_parameters_results_nuclear_sweep_15_1000'

file_name_list = [file_name_1, file_name_2, file_name_3, file_name_4]

for file in file_name_list:
    csv_file = file + '.csv'
    df_input_data = pd.read_csv(csv_file)
    hdf_file = file + '.h5'
    df_input_data.to_hdf(hdf_file, key = 'df_input_data')