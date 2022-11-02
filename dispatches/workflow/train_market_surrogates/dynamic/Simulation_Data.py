import os

__this_file_dir__ = os.getcwd()
import sys 
sys.path.append(__this_file_dir__)

from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from pyomo.environ import value, SolverFactory
from idaes.core.util import to_json, from_json
import time
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
            
            dispatch_data_fil: the file stores dispatch profiles by simulation years

            input_data_file: the file stores input data for parameter sweep

        Returns:
            
            list: [rt_dispatch_array, rt_lmp_array, da_dispatch_array, da_lmp_array]
        '''

        # read the data from excel by sheet names
        df_rt_dispatch = pd.read_excel(self.dispatch_data_file, sheet_name = 'rt_dispatch')
        df_rt_lmp = pd.read_excel(self.dispatch_data_file, sheet_name = 'rt_lmp')
        df_da_dispatch = pd.read_excel(self.dispatch_data_file, sheet_name = 'da_dispatch')
        df_da_lmp = pd.read_excel(self.dispatch_data_file, sheet_name = 'da_lmp')

        # drop the first column, which are indexes
        df_rt_dispatch_data = df_rt_dispatch.iloc[:, 1:]
        df_rt_lmp_data = df_rt_lmp.iloc[:, 1:]
        df_da_dispatch_data = df_da_dispatch.iloc[:, 1:]
        df_da_lmp_data = df_da_lmp.iloc[:, 1:]

        # the first column is the run_index. Put them in an array
        # indexes are the same for all sheets.
        run_index = df_rt_dispatch.iloc[:,0].to_numpy(dtype = str)

        # save the index in an list.
        # transfer from str to int and put them in a list
        index = []
        for run in run_index:
            index_num = re.split('_|\.',run)[1]
            index.append(int(index_num))

        # transfer the data to the np.array, dimension of test_years*8736(total hours in one simulation year)
        rt_dispatch_array = df_rt_dispatch_data.to_numpy(dtype = float)
        rt_lmp_array = df_rt_lmp_data.to_numpy(dtype = float)
        da_dispatch_array = df_da_dispatch_data.to_numpy(dtype = float)
        da_lmp_array = df_da_lmp_data.to_numpy(dtype = float)

        return [rt_dispatch_array, rt_lmp_array, da_dispatch_array, da_lmp_array], index


    def read_data_to_dict(self):
        
        '''
        Transfer the data into dictionary 
        
        Arguments: 
            
            dispatch_data_file: the file stores dispatch profiles by simulation years

            input_data_file: the file stores input data for parameter sweep

        Returns:
            
            dispatch_dict: {run_index:[dispatch data]}

            input_dict: {run_index:[input data]}
        '''

        data_list, index = self._read_data_to_array()

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

        # read the input data
        df_input_data = pd.read_hdf(self.input_data_file)
        # return the number of columns in the df, that is the dimension of the input space. 
        num_col = df_input_data.shape[1]

        # drop the first column, which is the indexes
        X = df_input_data.iloc[index,list(range(1,num_col))].to_numpy()

        input_data_dict = {}

        for num, x in zip(index, X):
            input_data_dict[num] = x

        self._dispatch_dict = rt_dispatch_dict
        self._input_data_dict = input_data_dict

        # put all the data in one dict
        data_dict = {}
        data_dict['rt_dispatch'] = rt_dispatch_dict
        data_dict['rt_lmp'] = rt_lmp_dict
        data_dict['da_dispatch'] = da_dispatch_dict
        data_dict['da_lmp'] = da_lmp_dict

        return data_dict, input_data_dict
    

    def _calculate_revenue(self):

        '''
        Calculate the revenue from the sweep data

        Arguments:

            None

        Return:

            rev_dict: dictionary that has the revenue data, {run_index: rev)}
        '''

        # the rt and da dispatch and lmp data are in data_list returned by self.read_data_to_dict
        data_dict, input_data_dict = self.read_data_to_dict()
        da_dispatch_dict = data_dict['da_dispatch']
        rt_dispatch_dict = data_dict['rt_dispatch']
        da_lmp_dict = data_dict['da_lmp']
        rt_lmp_dict= data_dict['rt_lmp']

        # get the run indexes
        index_list = list(self._dispatch_dict.keys())

        revenue_dict = {}
        for idx in index_list:
            da_dispatch_data_array = da_dispatch_dict[idx]
            da_lmp_data_array = da_lmp_dict[idx]
            rt_dispatch_data_array = rt_dispatch_dict[idx]
            rt_lmp_data_array = rt_lmp_dict[idx]

            revenue = 0
            for rt_dispatch, rt_lmp, da_dispatch, da_lmp in zip(da_dispatch_data_array, da_lmp_data_array, rt_dispatch_data_array, rt_lmp_data_array):
                # the revenue is equal to rt_lmp*(rt_dispatch - da_dispatch) + da_lmp*da_dispatch
                revenue += (rt_dispatch - da_dispatch)*rt_lmp + da_dispatch*da_lmp

            # revenue_dict = {run_index: revenue}
            revenue_dict[idx] = revenue

        return revenue_dict


    def _read_pmin(self):

        '''
        Read pmin from input_dict, this function is only for nuclear case study

        Arguments:
    
            dispatch_dict: dictionary stores dispatch data.

            input_dict: dictionary stores input data for parameter sweep

        Returns:
            pmin_dict: {run_index: pmin}
        '''

        if self.fixed_pmax == True:
            # the only nuclear generator in RTSGMLC, pmax = 400MW 
            pmax = 400
            index_list = list(self._dispatch_dict.keys())
            pmin_dict = {}

            for idx in index_list:
                # for NE sweep, the pmin_scaler is one of the swept parameters
                pmin_scaler = self._input_data_dict[idx][1]
                pmin_dict[idx] = pmax - pmax*pmin_scaler

        else:
            raise ValueError('For NE case study pmax must be fixed.')

        return pmin_dict


    def _read_pmax(self):

        '''
        Read pmax from input_dict according to the case study

        This is for RE and FE case study.

        Arguments:
    
            None

        Returns:

            pmax_dict: {run_index: pmax} 

            pmax: float, the max capacity.
        '''

        # if we sweep the pmax as input
        if self.fixed_pmax == False:
            # if the self.fixed_pmax == False, read the pmax from the input data
            # if the fixed_pmax = False, no difference between RE and FE.
            index_list = list(self._dispatch_dict.keys())

            # put the pmax in dictionary.
            pmax_dict = {}

            for idx in index_list:
                # if the parameter sweep is going to sweep pmax, we set it as the first element of the input data.
                pmax = self._input_data_dict[idx][0]
                pmax_dict[idx] = pmax

            return pmax_dict

        else:
            # when fix the pmax
            if self.case_type == 'RE':
                # now we only do parameter on WIND_303_1
                pmax = 847 # MW
                return pmax

            else:
                # FE case study, pmax = 436
                pmax = 436 # MW
                return pmax


    def _scale_data(self):

        '''
        scale the data by pmax to get capacity factors

        Arguments:

            None

        Returns:

            scaled_dispatch_dict: {run_index: [scaled dispatch data]}
        '''

        # for RE and FE, we can have varied pmax or fixex pmax. But the way we scale the data is the same. 
        if self.case_type in ['RE', 'FE']:
            
            if self.fixed_pmax == False:
                index_list = list(self._dispatch_dict.keys())

                pmax_dict = self._read_pmax()

                scaled_dispatch_dict = {}

                for idx in index_list:
                    dispatch_year_data = self._dispatch_dict[idx]
                    pmax_year = pmax_dict[idx]
                    # scale the data between [0,1]
                    scaled_dispatch_year_data = dispatch_year_data/pmax_year
                    scaled_dispatch_dict[idx] = scaled_dispatch_year_data
            else:
                index_list = list(self._dispatch_dict.keys())

                pmax = self._read_pmax()
                
                scaled_dispatch_dict = {}
                
                for idx in index_list:
                    dispatch_year_data = self._dispatch_dict[idx]
                    # scale the data between [0,1]
                    scaled_dispatch_year_data = dispatch_year_data/pmax
                    scaled_dispatch_dict[idx] = scaled_dispatch_year_data

        else:
            # NE case study use a different way to scale the data
            index_list = list(self._dispatch_dict.keys())
            pmin_dict = self._read_pmin()
            pmax = 400 # MW

            scaled_dispatch_dict = {}

            for idx in index_list:
                dispatch_year_data = self._dispatch_dict[idx]
                pmin_year = pmin_dict[idx]
                # scale the data between [0,1] where 0 is the Pmin (Pmax-Ppem)
                # this scale method is for only nuclear case study.
                scaled_dispatch_year_data = (dispatch_year_data - pmin_year)/(pmax - pmin_year)
                scaled_dispatch_dict[idx] = scaled_dispatch_year_data
        
        return scaled_dispatch_dict
