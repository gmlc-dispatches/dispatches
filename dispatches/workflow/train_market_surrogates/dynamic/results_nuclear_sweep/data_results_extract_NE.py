# extract_data
import pandas as pd
import os
import numpy as np

def extract_data(num_csv, sims_in_csv, folder_name):
	# direct to the path that has the every Prescient run results 
	current_path = os.getcwd()
	path1 = os.path.join(current_path, folder_name)

	# get all the file names in the folder, make it into a list.
	filenames = os.listdir(path1)
	# changes the current working directory to the given path
	os.chdir(path1)

	# shuffle the filenames to shuffle the simulation runs in each csv file
	# np.random.shuffle(filenames)

	size = sims_in_csv

	for i in range(num_csv):
		loop_index = []
		dispatch_frame = []
		for file in filenames[i*size:(i+1)*size]:
			# split the file name and get the index for the run as a list.
			index_ = file.split('_')[-1]
			loop_index.append('run_' + index_)
			df=pd.read_csv(file,dtype={"user_id": np.int16, "username": object})
			dispatch_data = df[['Dispatch']].T
			dispatch_frame.append(dispatch_data)

		column_name = df.index.astype(str)
		print(column_name)

		print(f'Generating Dispatch_data_NE_{folder_name}_{i}.csv')
		dispatch_output = pd.concat(dispatch_frame)
		dispatch_output.columns = column_name
		dispatch_output.index = loop_index
		csv_path = os.path.join(current_path, f"Dispatch_data_NE_{folder_name}_{i}.csv")
		dispatch_output.to_csv(csv_path, index=True)
		print(dispatch_output.info())
		print(f'Dispatch_data_NE_{folder_name}_{i}.csv completed...')


def main():
	num_csv = 1
	sims_in_csv = 48
	folder_name = 'results_nuclear_sweep_15_1000'
	extract_data(num_csv, sims_in_csv, folder_name)


if __name__ == "__main__":
	main()
