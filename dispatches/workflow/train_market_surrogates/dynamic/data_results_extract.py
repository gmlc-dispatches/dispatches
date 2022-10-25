# extract_data
import pandas as pd
import os
import numpy as np
import re

def extract_data_seperately(folder_name, num_sims, case_name):

	'''
	Extract only dispatch data from one folder. 

	Use this function when we do not want to put Prescient variables into the input space

	Arguments:

		num_sims: int, number of simulations you want to have in the csv.

		folder_name: str, name of the folder that has the data we want to collect.

		case_name: str, name of the case study.

	Returns:

		None
	'''

	# set the current path as the 'root_path'
	root_path = os.getcwd()

	# path1 is the folder with simulations under different Prescient variables.
	path1 = os.path.join(root_path, 'results_renewable_sweep_Wind_H2', folder_name)

	# reorganize the filename by index. Otherwise the simulation range in the csv will not be from 0 to num_sims-1
	new_filenames = []
	for i in range(num_sims):
		new_name = f'sweep_results_index_{i}.csv'
		new_filenames.append(new_name)
	
	# changes the working directory to the given path
	os.chdir(path1)	

	# set two empty lists to store index and dispatch data
	loop_index = []
	dispatch_frame = []

	for file in new_filenames:
		# split the file name and get the index for the run as a list.
		index_ = int(re.split('[_.]', file)[-2])
		loop_index.append('run_' + str(index_))
		df=pd.read_csv(file,dtype={"user_id": np.int16, "username": object})
		# get the dispatch data
		dispatch_data = df[['Dispatch']].T
		dispatch_frame.append(dispatch_data)
	
	# set the column names as string type
	column_name = df.index.astype(str)
	print(f'Generating Dispatch_data_{case_name}_{folder_name}.csv')
	dispatch_output = pd.concat(dispatch_frame)
	# set column and row index for the csv
	dispatch_output.columns = column_name
	dispatch_output.index = loop_index
	csv_path = os.path.join(root_path, f"results_renewable_sweep_Wind_H2\\Dispatch_data_{case_name}_{folder_name}.csv")
	print(csv_path)
	dispatch_output.to_csv(csv_path, index=True)
	print(dispatch_output.info())
	print(f'Dispatch_data_{case_name}_{folder_name}.csv completed...')

	# go back to the root path
	os.chdir(root_path)

	return

def extract_data_whole(num_sims, folder_name_list, case_name):

	'''
	Extract dispatch data from the folder list and make them in one csv file. 

	Use this function when we do want to put Prescient variables into the input space.

	Please becareful about the index.

	Arguments:

		num_sims: int, number of simulations you want to have in the csv.

		folder_name_list: list, name of the folder that has the data we want to collect.

		case_name: str, name of the case study.

	Returns:

		None
	'''
	root_path = os.getcwd()
	size = num_sims
	loop_index = []
	dispatch_frame = []

	for idx, folder_name in enumerate(folder_name_list):
		# direct to the path that has the every Prescient run results 	
		path1 = os.path.join(root_path, 'results_renewable_sweep_Wind_H2', folder_name)

		# reorganize the filename by index
		new_filename = []
		for i in range(56):
			new_name = f'sweep_results_index_{i}.csv'
			new_filename.append(new_name)
		# changes the current working directory to the given path
		os.chdir(path1)


		for file in new_filename:
			# split the file name and get the index for the run as a list.
			index_ = int(re.split('[_.]', file)[-2]) + idx*56
			loop_index.append('run_' + str(index_))
			df=pd.read_csv(file,dtype={"user_id": np.int16, "username": object})
			dispatch_data = df[['Dispatch']].T
			dispatch_frame.append(dispatch_data)

			column_name = df.index.astype(str)

		os.chdir(root_path)

	print(f'Generating Dispatch_data_{case_name}_whole.csv')
	dispatch_output = pd.concat(dispatch_frame)
	dispatch_output.columns = column_name
	dispatch_output.index = loop_index
	result_path = os.path.join(root_path,f'results_renewable_sweep_Wind_H2\\Dispatch_data_{case_name}_whole.csv')
	dispatch_output.to_csv(result_path, index=True)
	print(dispatch_output.info())
	print(f'Dispatch_data_{case_name}_whole.csv completed...')
	# pd.options.display.max_rows = None
	# print(dispatch_output)


def main():
	# need to change the num_sims if you wand to do extract_data_whole

	num_sims = 56

	case_name = 'RE_H2'
	
	f1 = 'results_parameter_sweep_10_500'
	f2 = 'results_parameter_sweep_10_1000'
	f3 = 'results_parameter_sweep_15_500'
	f4 = 'results_parameter_sweep_15_1000'
	folder_name_list = [f1,f2,f3,f4]

	for folder in folder_name_list:
		extract_data_seperately(folder, num_sims, case_name)

	extract_data_whole(56*4, folder_name_list, 'RE_H2')


if __name__ == "__main__":
	main()
