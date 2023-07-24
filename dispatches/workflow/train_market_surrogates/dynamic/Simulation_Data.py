#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

import pandas as pd
import numpy as np
import re
import pathlib
from dispatches_data.api import path


class SimulationData:
    def __init__(self, dispatch_data_file, input_data_file, num_sims, case_type):

        '''
        Initialization for the class
        
        Arguments:
            
            dispatch_data_file: data path that has the dispatch data
            
            input_data_file: data path that has the input data for sweep

            num_sims: int, number of simulations that we are going to read.

            case_type: str, must be one of 'RE, NE, FE'
        
        Returns:

            None
        '''
        
        self.dispatch_data_file = dispatch_data_file
        self.input_data_file = input_data_file
        self.num_sims = num_sims
        self.case_type = case_type
        self.read_data_to_dict()

        # default rt wind file
        path_to_data_package = path("dynamic_sweep")
        self.default_rt_wind_file = path_to_data_package / "RE" / "Real_Time_wind_hourly.csv"
        

    @property
    def num_sims(self):

        '''
        Porperty getter of num_sims

        Returns:

            int: number of years for clustering, positive integer.
        '''

        return self._num_sims


    @num_sims.setter
    def num_sims(self, value):

        '''
        Property setter of num_sims

        Arguments:

            value: intended value for num_sims 

        Returns:
            
            None
        '''

        if not isinstance(value, int):
            raise TypeError(
                f"The number of clustering years must be positive integer, but {type(value)} is given."
            )

        if value < 1:
            raise ValueError(
                f"The number of simulation years must be positive integer, but {value} is given."
            )

        self._num_sims = value


    @property
    def case_type(self):

        '''
        Porperty getter of case_type

        Returns:

            str: the case study type.
        '''

        return self._case_type


    @case_type.setter
    def case_type(self, value):

        '''
        Property setter of case_type

        Arguments:

            value: intended value for case_type

        Returns:
            
            None
        '''

        if not isinstance(value, str):
            raise TypeError(
                f"The value of case_type must be str, but {type(value)} is given."
            )

        if value not in ['RE','NE','FE']:
            raise ValueError(
                f"The case_type must be one of 'RE','NE' or 'FE', but {value} is given."
            )

        self._case_type = value


    def _read_data_to_array(self):

        '''
        Read the dispatch data from the csv file

        Arguments:
            
            None

        Returns:
            
            list: [rt_dispatch_array, rt_lmp_array, da_dispatch_array, da_lmp_array]
        '''

        # read the data from excel by sheet names
        df_dispatch = pd.read_csv(self.dispatch_data_file, nrows = self.num_sims)

        # drop the first column, which are indexes
        df_dispatch_data = df_dispatch.iloc[:, 1:]

        # the first column is the run_index. Put them in an array
        # indexes are the same for all sheets.
        run_index = df_dispatch.iloc[:,0].to_numpy(dtype = str)

        # save the index in an list.
        # transfer from str to int and put them in a list
        index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            index.append(int(index_num))

        # transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
        dispatch_array = df_dispatch_data.to_numpy(dtype = float)

        return dispatch_array, index


    def read_data_to_dict(self):
        
        '''
        Transfer the data into dictionary 
        
        Arguments: 
            
            None

        Returns:
            
            dispatch_dict: {run_index:[dispatch data]}

            input_dict: {run_index:[input data]}
        '''

        dispatch_array, index = self._read_data_to_array()

        # put all the data in a dict
        dispatch_dict = {}

        # the dict will be {run_index: data}
        for num, idx in enumerate(index):
            dispatch_dict[idx] = dispatch_array[num]

        # read the input data
        df_input_data = pd.read_hdf(self.input_data_file)
        # return the number of columns in the df, that is the dimension of the input space. 
        num_col = df_input_data.shape[1]

        # drop the first column, which is the indexes
        X = df_input_data.iloc[index,list(range(1,num_col))].to_numpy()

        input_data_dict = {}

        for num, x in zip(index, X):
            input_data_dict[num] = x

        # save te dispatch_dict, input_dict and index in the class property
        self._dispatch_dict = dispatch_dict
        self._input_data_dict = input_data_dict
        self._index = index

        return dispatch_dict, input_data_dict


    def _read_NE_pmin(self):

        '''
        Read pmin from input_dict, this function is only for nuclear case study

        Arguments:

            None

        Returns:
            pmin_dict: {run_index: pmin}
        '''

        # the only nuclear generator in RTSGMLC, pmax = 400MW 
        pmax = 400
        pmin_dict = {}

        for idx in self._index:
            # for NE sweep, the pmin_scaler is one of the swept parameters
            pmin_scaler = self._input_data_dict[idx][1]
            pmin_dict[idx] = pmax - pmax*pmin_scaler

        return pmin_dict


    def _read_RE_pmax(self, wind_gen = '303_WIND_1'):

        '''
        Read the pmax for RE case study

        Arguments:

            wind_gen: The name of the wind generator, default '303_WIND_1'

        Returns:

            pmax: float, the pmax of the wind generator.
        '''
        # make a dict key = generator name, value = pmax
        _wind_gen_pmax = {}
        _wind_gen_name = ['309_WIND_1', '317_WIND_1', '303_WIND_1', '122_WIND_1']
        _win_gen_pmax_list = [148.3, 799.1, 847.0, 713.5]


        for name, pmax in zip(_wind_gen_name, _win_gen_pmax_list):
            _wind_gen_pmax[name] = pmax

        # check the wind generator and pv generator name are correct.
        # Assign pmax accroding to generators.
        if wind_gen in _wind_gen_name:
            wind_gen_pmax = _wind_gen_pmax[wind_gen]
        else:
            raise NameError("wind generator name {} is invaild.".format(wind_gen))   
        
        return wind_gen_pmax


    def _read_FE_pmax(self):

        '''
        Read the pmax for FE case study

        Arguments:

            None

        Returns:

            pmax_dict: dict, the pmax of the generator, {run_index: pmax}.
        '''
        # pmax of the generator is 436
        pmax = 436

        # put the pmax in dictionary.
        pmax_dict = {}

        for idx in self._index:
            # the third elemet of the input data is the storage size
            p_storage = self._input_data_dict[idx][1]
            pmax_dict[idx] = pmax + p_storage

        return pmax_dict


    def _scale_data(self):

        '''
        scale the data by pmax to get capacity factors

        Arguments:

            None

        Returns:

            scaled_dispatch_dict: {run_index: [scaled dispatch data]}
        '''

        # for FE, we should scale the data by pmax+storage
        if self.case_type == 'FE':
            pmax_dict = self._read_FE_pmax()
            # store the scaled data in the dict
            scaled_dispatch_dict = {}

            for idx in self._index:
                dispatch_year_data = self._dispatch_dict[idx]
                pmax_year = pmax_dict[idx]
                # scale the data, for FE, when storage is deployed, cf is higher than 1.
                scaled_dispatch_year_data = (dispatch_year_data-284)/(436-284)

                # for every time period in the scaled year
                for i,c in enumerate(scaled_dispatch_year_data):
                    # c > 1 means storage depolyed. Scale the data between [1,1.2]
                    if c > 1:
                        scaled_dispatch_year_data[i] = ((scaled_dispatch_year_data[i]-1)*(436-284))/(pmax_dict[idx]-436)*0.2 + 1

                scaled_dispatch_dict[idx] = scaled_dispatch_year_data

        if self.case_type == 'NE':
            # NE case study use a different way to scale the data
            pmin_dict = self._read_NE_pmin()
            pmax = 400 # MW
            # store the scaled data in the dict
            scaled_dispatch_dict = {}

            for idx in self._index:
                dispatch_year_data = self._dispatch_dict[idx]
                pmin_year = pmin_dict[idx]
                # scale the data between [0,1] where 0 is the Pmin (Pmax-Ppem)
                # this scale method is for only nuclear case study.
                scaled_dispatch_year_data = (dispatch_year_data - pmin_year)/(pmax - pmin_year)
                scaled_dispatch_dict[idx] = scaled_dispatch_year_data
        
        if self.case_type == 'RE':
            # fixed pmax, should be a scalar
            pmax = self._read_RE_pmax()
            # store the scaled data in the dict
            scaled_dispatch_dict = {}
            
            for idx in self._index:
                dispatch_year_data = self._dispatch_dict[idx]
                # scale the data between [0,1] by pmax
                scaled_dispatch_year_data = dispatch_year_data/pmax
                scaled_dispatch_dict[idx] = scaled_dispatch_year_data

        return scaled_dispatch_dict


    def read_wind_data(self):
        
        '''
        read the wind data (from RTS_GMLC)

        Defaut wind gen is '303_WIND_1', pmax = 847

        Arguments:

            wind_file: path of the wind data

        Returns:

            wind_data: list, wind data
        '''

        # for '303_WIND_1', pmax = 847MW
        wind_gen = '303_WIND_1'
        wind_gen_pmax = 847  # MW

        # use the default wind rt file from RTS-GMLC
        wind_file = self.default_rt_wind_file
        total_wind_profile = pd.read_csv(wind_file)
        selected_wind_data = total_wind_profile[wind_gen].to_numpy()

        selected_wind_data = selected_wind_data/wind_gen_pmax

        wind_data = []
        time_len = 24
        day_num = int(len(selected_wind_data)/time_len)
        for i in range(day_num):
            wind_data.append(np.array(selected_wind_data[i*24:(i+1)*24]))

        # wind_data will have shape of (364, 24) with all data scaled by p_wind_max
        return wind_data



    def read_rev_data(self, rev_path):

        '''
        Calculate the revenue from the sweep data

        Arguments:

            rev_path: the path of the revenue data

        Return:

            rev_dict: dictionary that has the revenue data, {run_index: rev)}
        '''

        # read the revenue data from the csv. Keep nrows same with the number of simulations. 
        df_rev = pd.read_csv(rev_path, nrows = self.num_sims)
        # drop the first col, indexes.
        rev_array = df_rev.iloc[:, 1:].to_numpy(dtype = float)

        # get the run indexes
        index_list = list(self._dispatch_dict.keys())

        revenue_dict = {}
        for i, idx in enumerate(index_list):
            revenue_dict[idx] = rev_array[i][0]

        return revenue_dict