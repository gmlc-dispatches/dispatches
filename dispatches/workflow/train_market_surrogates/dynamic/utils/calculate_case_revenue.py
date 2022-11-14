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

import pandas as pd
import os
import numpy as np
import re


def read_data_to_array(csv_name_list):

    '''
    Read the dispatch data from the csv file

    Arguments:
        
        dispatch_data_fil: the file stores dispatch profiles by simulation years

        input_data_file: the file stores input data for parameter sweep

    Returns:
        
        list: [rt_dispatch_array, rt_lmp_array, da_dispatch_array, da_lmp_array]
    '''

    # read the data from csv by sheet names

    target_data = []
    for csv_name in csv_name_list:
        df_data = pd.read_csv(csv_name)


        # drop the first column, which are indexes
        df_useful_data = df_data.iloc[:, 1:]


        # the first column is the run_index. Put them in an array
        # indexes are the same for all sheets.
        run_index = df_data.iloc[:,0].to_numpy(dtype = str)

        # save the index in an list.
        # transfer from str to int and put them in a list
        index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            index.append(int(index_num))

        # transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
        df_data_array = df_useful_data.to_numpy(dtype = float)
        target_data.append(df_data_array)

    return target_data, index


def read_data_to_dict(csv_name_list):
    
    '''
    Transfer the data into dictionary 
    
    Arguments: 
        
        dispatch_data_file: the file stores dispatch profiles by simulation years

        input_data_file: the file stores input data for parameter sweep

    Returns:
        
        dispatch_dict: {run_index:[dispatch data]}

        input_dict: {run_index:[input data]}
    '''

    data_list, index = read_data_to_array(csv_name_list)

    # put all the data in a dict
    rt_dispatch_dict = {}
    rt_lmp_dict = {}
    da_dispatch_dict = {}
    da_lmp_dict = {}

    # the dict will be {run_index: data}
    for num, idx in enumerate(index):
        rt_dispatch_dict[idx] = data_list[0][num]
        rt_lmp_dict[idx] = data_list[1][num]
        da_dispatch_dict[idx] = data_list[2][num]
        da_lmp_dict[idx] = data_list[3][num]

    # put all the data in one dict
    data_dict = {}
    data_dict['rt_dispatch'] = rt_dispatch_dict
    data_dict['rt_lmp'] = rt_lmp_dict
    data_dict['da_dispatch'] = da_dispatch_dict
    data_dict['da_lmp'] = da_lmp_dict

    return data_dict


def calculate_revenue(csv_name_list, result_path):

    '''
    Calculate the revenue from the sweep data

    Arguments:

        None

    Return:

        rev_dict: dictionary that has the revenue data, {run_index: rev)}
    '''

    # the rt and da dispatch and lmp data are in data_list returned by self.read_data_to_dict
    data_dict = read_data_to_dict(csv_name_list)
    da_dispatch_dict = data_dict['da_dispatch']
    rt_dispatch_dict = data_dict['rt_dispatch']
    da_lmp_dict = data_dict['da_lmp']
    rt_lmp_dict= data_dict['rt_lmp']

    # get the run indexes
    index_list = list(da_dispatch_dict.keys())

    revenue_dict = {}
    for idx in index_list:
        da_dispatch_data_array = da_dispatch_dict[idx]
        da_dis = np.isnan(da_dispatch_data_array).any()    # do not calculate nan data
        da_lmp_data_array = da_lmp_dict[idx]
        da_lmp = np.isnan(da_lmp_data_array).any() 
        rt_dispatch_data_array = rt_dispatch_dict[idx]
        rt_dis = np.isnan(rt_dispatch_data_array).any() 
        rt_lmp_data_array = rt_lmp_dict[idx]
        rt_lmp = np.isnan(rt_lmp_data_array).any()

        # if no np.nan in the data, it will be not false.
        if not (da_dis or da_lmp or rt_dis or rt_lmp):
            revenue = 0
            for rt_dispatch, rt_lmp, da_dispatch, da_lmp in zip(da_dispatch_data_array, da_lmp_data_array, rt_dispatch_data_array, rt_lmp_data_array):
                # the revenue is equal to rt_lmp*(rt_dispatch - da_dispatch) + da_lmp*da_dispatch
                revenue += (rt_dispatch - da_dispatch)*rt_lmp + da_dispatch*da_lmp

            # revenue_dict = {run_index: revenue}
            revenue_dict[idx] = revenue

    df_rev = pd.Series(revenue_dict)
    df_rev.to_csv(result_path, index=True)

    return revenue_dict
