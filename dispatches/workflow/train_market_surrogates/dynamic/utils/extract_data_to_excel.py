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

# This script can extract the rt_dispatch, rt_lmp, da_dispatch, da_lmp to one excel file with 4 sheets.
import pandas as pd
import numpy as np
import re
import os
import pathlib

def extract_to_excel(data_path, folder_name_list, case_name): 

    '''
    Extract the data to an excel file that contains 4 sheets (RT_dispatch, RT_LMP, DA_dispatch, DA_LMP.)
    
    Arguments:
        
        data_path: str, the folder path that contains the sweep results. (all sweeps in different Prescient parameters)

        folder_name_list: list, name of the folder that has the data we want to collect.

        case_name: str, name of the case study.

    Returns


    '''

    root_path = pathlib.Path.cwd()
    loop_index = []
    rt_dispatch = []
    rt_lmp = []
    da_dispatch = []
    da_lmp = []
    num_csv_list = []

    for folder_name in folder_name_list:
        # generate path that has the extact csv files
        path1 = str(pathlib.Path.joinpath(root_path,data_path, folder_name))

        # read the number of files in the folder
        # num_csv = len(os.listdir(path1))
        num_csv = 50

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

            # read real time dispatch data
            rt_dispatch_data = df[['Dispatch']].T
            rt_dispatch.append(rt_dispatch_data)

            # read real time lmp data
            rt_lmp_data = df[['LMP']].T
            rt_lmp.append(rt_lmp_data)

            # read the day ahead dispatch data
            da_dispatch_data = df[['Dispatch DA']].T
            da_dispatch.append(da_dispatch_data)

            # read the day ahead lmp data
            da_lmp_data = df[['LMP DA']].T
            da_lmp.append(da_lmp_data)
            

            column_name = df.index.astype(str)

        num_csv_list.append(num_csv)
        os.chdir(root_path)

    # specify the file name and writer
    excel_name = f'Dispatch_data_{case_name}_whole.xlsx'
    writer = pd.ExcelWriter(excel_name)

    print('Generating', excel_name)
    # concat the dataframe and specify indexes
    rt_dispatch_output = pd.concat(rt_dispatch)
    rt_dispatch_output.index = loop_index
    rt_lmp_output = pd.concat(rt_lmp)
    rt_lmp_output.index = loop_index
    da_dispatch_output = pd.concat(da_dispatch)
    da_dispatch_output.index = loop_index
    da_lmp_output = pd.concat(da_lmp)
    da_lmp_output.index = loop_index

    # write to excel in different sheets
    rt_dispatch_output.to_excel(writer, sheet_name = 'rt_dispatch', index = True)
    rt_lmp_output.to_excel(writer, sheet_name = 'rt_lmp', index = True)
    da_dispatch_output.to_excel(writer, sheet_name = 'da_dispatch', index = True)
    da_lmp_output.to_excel(writer, sheet_name = 'da_lmp', index = True)
    writer.save()
    print(excel_name, 'completed...')

    return 

