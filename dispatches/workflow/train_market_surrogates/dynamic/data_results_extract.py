# extract_data
import pandas as pd
import os
import numpy as np


def main():
	path = os.getcwd()
	path1 = path + '\\idaes_parameter_sweep_results_0'
	filenames = os.listdir(path1)
	os.chdir(path1)

	# print(filenames[0:10])
	loop_index = []
	LMP_frame = []
	dispatch_frame = []

	for file in filenames[:100]:
		index_ = file.split('_')[-1]
		loop_index.append('run_' + index_)
		df=pd.read_csv(file,dtype={"user_id": np.int16, "username": object})
		LMP_data = df[['LMP']].T
		LMP_frame.append(LMP_data)
		dispatch_data = df[['Dispatch']].T
		dispatch_frame.append(dispatch_data)

	column_name = df.index.astype(str)
	print(column_name)
	print('Generating LMP_test.csv...')
	LMP_output = pd.concat(LMP_frame)
	LMP_output.columns = column_name
	LMP_output.index = loop_index
	LMP_output.to_csv(os.path.join(path, "Example_LMP_test.csv"), index=True)
	print(LMP_output.info())
	print('LMP_test.csv completed...')

	print('Generating dispatch_test.csv...')
	dispatch_output = pd.concat(dispatch_frame)
	dispatch_output.columns = column_name
	dispatch_output.index = loop_index
	dispatch_output.to_csv(os.path.join(path, "Example_dispatch_test.csv"), index=True)
	print(dispatch_output.info())
	print('dispatch_test.csv completed...')

if __name__ == "__main__":
	main()
