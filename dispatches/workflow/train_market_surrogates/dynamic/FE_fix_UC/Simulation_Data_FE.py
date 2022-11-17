import os

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from idaes.core.util import to_json, from_json
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt


# need to find out a place to store the data instead of just put them in the dispatches repo
# temporarily put them in the repo


class SimulationData:
    def __init__(self, dispatch_data_file, input_data_file, num_sims, case_type, fixed_pmax = True):

        '''
        Initialization for the class
        
        Arguments:
            
            dispatch_data_file: data path that has the dispatch data
            
            input_data_file: data path that has the input data for sweep

            num_sims: int, number of simulations that we are going to read.

            case_type: str, must be one of 'RE, NE, FE'

            fixed_pmax: bool, default True. If the pmax of the generator is fixed. 
        
        Returns:

            None
        '''
        
        self.dispatch_data_file = dispatch_data_file
        self.input_data_file = input_data_file
        self.num_sims = num_sims
        self.case_type = case_type
        self.fixed_pmax = fixed_pmax
        self.read_data_to_dict()


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


    @property
    def fixed_pmax(self):

        '''
        Porperty getter of fixed_pmax

        Returns:

            bool: the fixed_pmax bool
        '''

        return self._fixed_pmax


    @fixed_pmax.setter
    def fixed_pmax(self, value):

        '''
        Property setter of fixed_pmax

        Arguments:

            value: intended value for fixed_pmax

        Returns:
            
            None
        '''

        if not isinstance(value, bool):
            raise TypeError(
                f"The fixed_pmax must be bool, but {type(value)} is given."
            )

        self._fixed_pmax = value


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


    def _read_FE_pmax(self):

        '''
        Read the pmax for FE case study

        Arguments:

            None

        Returns:

            None
        '''
        # pmax of the generator is 436
        pmax = 436
        self.pmax = pmax

        return pmax


    def _read_FE_pmax_dict(self):

        '''
        Read the pmax for FE case study

        Arguments:

            None

        Returns:

            None
        '''
        # pmax of the generator is 436
        pmax = 436
        pmax_dict = {}
        for i in self._index:
            ps = self._input_data_dict[i][1]
            pmax_dict[i] = ps+pmax

        return pmax_dict


    def _scale_data(self):

        '''
        scale the data by pmax to get capacity factors

        Arguments:

            None

        Returns:

            scaled_dispatch_dict: {run_index: [scaled dispatch data]}
        '''
        # pmax = self._read_FE_pmax()
        pmax_dict = self._read_FE_pmax_dict()
        scaled_dispatch_dict = {}

        for idx in self._index:
            dispatch_year_data = self._dispatch_dict[idx]
            # scale the data between [0,1]
            scaled_dispatch_year_data = (dispatch_year_data-284)/(436-284)
            
            # scale capacity factor > 1 
            for i,c in enumerate(scaled_dispatch_year_data):
                if c > 1:
                    # for different scale method,  
                    scaled_dispatch_year_data[i] = ((scaled_dispatch_year_data[i]-1)*(436-284))/(pmax_dict[idx]-436)*0.2 + 1

            scaled_dispatch_dict[idx] = scaled_dispatch_year_data

        return scaled_dispatch_dict


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