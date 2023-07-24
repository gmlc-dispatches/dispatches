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
import numpy as np
import os


def read_seperately(file_name):

    csv_file = file_name
    df_input_data = pd.read_csv(csv_file)

    return df_input_data


def read_together(file_name_list, case_type, num_sims, result_path):
    # num_sims is the number of simulations under the fixed reserve_factor and max_lmp
    rf = [10,10,15,15]
    mlmp = [500,1000,500,1000]


    df_input_data_list = []
    for reserve_factor, max_lmp, file_name in zip(rf, mlmp, file_name_list):
        df_input_data = read_seperately(file_name)

        reserve_factor_array = np.array([reserve_factor]*num_sims)
        max_lmp_array = np.array([max_lmp]*num_sims)

        df_input_data.insert(df_input_data.shape[1],'reserve_factor', reserve_factor_array)
        df_input_data.insert(df_input_data.shape[1],'max_lmp', max_lmp_array)
        df_input_data_list.append(df_input_data)

    for idx, df in enumerate(df_input_data_list):
        for j in range(num_sims):
            df.loc[j,'index'] = j+(idx)*num_sims

    df_all = pd.concat(df_input_data_list, axis=0, ignore_index=True)

    print(f'Generating {result_path}')
    df_all.to_hdf(result_path, key = 'df_all')
    # pd.options.display.max_rows = None
    # print(df_all)

    return
