# extract_data to excel
import pandas as pd
import os
import numpy as np
import re

# 4 sheets, corresponding to RT_dispatch, RT_LMP, DA_dispatch, DA_LMP.

def extract_to_excel(data_path, folder_name_list, case_name): 

    '''
    Extract the data to an excel file that contains 4 sheets (RT_dispatch, RT_LMP, DA_dispatch, DA_LMP.)
    
    Arguments:
        
        data_path: str, the folder path that contains the sweep results. (all sweeps in different Prescient parameters)

        folder_name_list: list, name of the folder that has the data we want to collect.

        case_name: str, name of the case study.

    Returns


    '''

    root_path = os.getcwd()
    loop_index = []
    rt_dispatch = []
    rt_lmp = []
    da_dispatch = []
    da_lmp = []
    num_csv_list = []

    for folder_name in folder_name_list:
        # generate path that has the extact csv files
        path1 = os.path.join(root_path, data_path, folder_name)

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


def main():
    # need to change the num_sims if you wand to do extract_data_whole

    case_name = 'NE'
    
    # f1 = 'results_parameter_sweep_10_500'
    # f2 = 'results_parameter_sweep_10_1000'
    # f3 = 'results_parameter_sweep_15_500'
    # f4 = 'results_parameter_sweep_15_1000'

    f1 = 'results_nuclear_sweep_10_500'
    f2 = 'results_nuclear_sweep_10_1000'
    f3 = 'results_nuclear_sweep_15_500'
    f4 = 'results_nuclear_sweep_15_1000'
    folder_name_list = [f1,f2,f3,f4]

    data_path = 'results_nuclear_sweep'

    extract_to_excel(data_path, folder_name_list, case_name)


if __name__ == "__main__":
    main()