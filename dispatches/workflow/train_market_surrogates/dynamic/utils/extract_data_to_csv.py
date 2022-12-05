#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################

# extract_data
import pandas as pd
import numpy as np
import re
import os
import pathlib

# Extract RT_dispatch, RT_LMP, DA_dispatch, DA_LMP data to different csv files.
def extract_data(data_path, folder_name_list, case_name, data_name): 

    '''
    Extract the data to an csv file.
    
    Arguments:
        
        data_path: str, the folder path that contains the sweep results. (all sweeps in different Prescient parameters)

        folder_name_list: list, name of the folder that has the data we want to collect.

        case_name: str, name of the case study.

        data_name: 'Dispatch', 'LMP', 'Dispatch DA', 'LMP DA'

    Returns


    '''

    root_path = pathlib.Path.cwd()
    loop_index = []
    data_list = []
    num_csv_list = []

    for folder_name in folder_name_list:
        # generate path that has the extact csv files
        path1 = str(pathlib.Path.joinpath(root_path, data_path, folder_name))

        # read the number of files in the folder
        num_csv = len(os.listdir(path1))

        # reorganize the filename by index
        new_filename = []
        for i in range(num_csv):
            new_name = f'sweep_results_index_{i}.csv'
            new_filename.append(new_name)
        # changes the current working directory to the given path
        os.chdir(path1)

        for file in new_filename:
            # split the file name and get the index for the run as a list.
            index_ = int(re.split('[_.]', file)[-2]) + sum(num_csv_list)
            loop_index.append('run_' + str(index_))
            df=pd.read_csv(file,dtype={"user_id": np.int16, "username": object})

            # read data
            target_data = df[[data_name]].T
            data_list.append(target_data)
            
            column_name = df.index.astype(str)

        num_csv_list.append(num_csv)
        os.chdir(root_path)

    # specify the file name and writer
    csv_name = f'Dispatch_data_{case_name}_{data_name}_whole.csv'

    print('Generating', csv_name)
    # concat the dataframe and specify indexes
    data_output = pd.concat(data_list)
    data_output.index = loop_index
    data_output.columns = column_name
    # write to csv in different sheets
    data_out_path = pathlib.Path.joinpath(root_path ,csv_name)
    data_output.to_csv(data_out_path, index=True)

    print(csv_name, 'completed...')

    return