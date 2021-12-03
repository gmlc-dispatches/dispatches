#This is a stripped-down version of Xian's analysis code used to analyze results of individual Prescient simulations
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import json

################################################################################
############################ Helper functions ##################################
################################################################################

def df_col_to_dict(df,key_col_name,value_col_name):
    '''
    A funtion to assemble a dictionary, who has keys 'key_col_name' and values
    'value_col_name' from a pandas dataframe.

    Arguments:
        key_col_name: the column which will be used as the key of the dict [str]
        value_col_name: the column which will be used as the value of the dict [str]

    Return
        the new dict
    '''
    return dict(zip(df[key_col_name],df[value_col_name]))

def get_data_given(df, bus=None, date=None, hour=None, generator=None, fuel_type=None):

    '''
    This function gets the data out of a pandas dataframe given one or more
    options, e.g. time.

    Arguments:
        df: the dataframe we are interested in
        bus: the bus ID we want [int]
        date: the date we want [str]
        hour: the hour we want [int]
        generator: the generator ID [str]
        fuel_type: generator fuel, e.g. Coal [str]
    Returns:
        df: a dataframe that has the information we specified.
    '''

    # get data given bus id
    if bus is not None:
        # in the original rts-gmlc dataset there is a Bus ID col
        if 'Bus ID' in df.columns:
            df = df.loc[(df['Bus ID'] == bus)]

        # in the prescient result data we have to extract bus id from other col
        # e.g. gennerator name col
        elif 'Generator' in df.columns:
            # convert the type to str
            bus = str(bus)
            # find the rows that starts with the bus name
            searchrows = df['Generator'].str.startswith(bus)
            df = df.loc[searchrows,:]

    # get data given date
    if date is not None:
        df = df.loc[(df['Date'] == date)]

    # get data given hour
    if hour is not None:
        df = df.loc[(df['Hour'] == hour)]

    # get data given hour
    if generator is not None:

        # Similarly this is for Prescient result data
        if 'Generator' in df.columns:
            df = df.loc[(df['Generator'] == generator)]
        # this is for rts-gmlc dataset
        elif 'GEN UID' in df.columns:
            df = df.loc[(df['GEN UID'] == generator)]

    # get data given fuel
    if fuel_type is not None:
        df = df.loc[df['Fuel'] == fuel_type]

    return df

def sum_data_to_dict(df,node_edge_list,data_col):
    '''
    A function to calculate the sum value at a specific bus/edge. For example,
    the total capacity at a bus.

    Arguments:
        df: the dataframe that has the results
        node_edge_list: a list of all the nodes or edges you interested in
        data_col: the data column you want (e.g. 'Dispatch')

    Returns:
        the resulted dictionary
    '''

    # assemble sums into a dict
    result = {i: get_data_given(df,bus = i)[data_col].sum() for i in node_edge_list}

    return result

class NetworkData:

    def __init__(self,network_data_dir = '../RTS-GMLC/RTS_Data/SourceData/'):
        '''
        This class reads in RTS-GMLC dataset and makes network plots.

        Arguments:
            network_data_dir: directory containing RTS-GMLC raw data [string]
        '''

        assert isinstance(network_data_dir, str), "Arugment network_data_dir is the wrong type! It should a string."

        # read the bus and branch data
        self.bus_df = pd.read_csv(network_data_dir + 'bus.csv')
        self.branch_df = pd.read_csv(network_data_dir + 'branch.csv')

        # generator params (this has the capacity of each generator)
        self.gen_param_df = pd.read_csv(network_data_dir + 'gen.csv')

        # a dictionary that maps bus id to bus name
        self.bus_id_to_bus_name = df_col_to_dict(self.bus_df,'Bus ID','Bus Name')
        self.bus_name_to_bus_id = df_col_to_dict(self.bus_df,'Bus Name','Bus ID')

        # thermal generators df
        dispatchable_fuel_types = ['Coal','Oil','NG','Nuclear']
        self.thermal_gen_param_df = self.gen_param_df.loc[self.gen_param_df['Fuel'].isin(dispatchable_fuel_types)]

        # renewable generators df
        renewable_fuel_types = ['Hydro','Solar','Wind']
        self.renewable_gen_param_df = self.gen_param_df.loc[self.gen_param_df['Fuel'].isin(renewable_fuel_types)]

        # bus id
        self.bus_id = list(self.bus_df['Bus ID'])

        self.num_buses = len(self.bus_id)
        self.num_thermal_generators = len(self.thermal_gen_param_df)
        self.num_renewable_generators = len(self.renewable_gen_param_df)
        self.total_num_generator = self.num_thermal_generators + self.num_renewable_generators
        self.total_thermal_power = self.thermal_gen_param_df['PMax MW'].sum()
        self.total_renewable_power = self.renewable_gen_param_df['PMax MW'].sum()
        self.total_power = self.gen_param_df['PMax MW'].sum()

        self.bus_thermal_pmax_dict = sum_data_to_dict(self.thermal_gen_param_df,\
        self.bus_id,data_col = 'PMax MW')

        # bus renewable pmax dict
        self.bus_renewable_pmax_dict = sum_data_to_dict(self.renewable_gen_param_df,\
        self.bus_id,data_col = 'PMax MW')

        # bus total pmax dict
        self.bus_pmax_dict = sum_data_to_dict(self.gen_param_df,\
        self.bus_id,data_col = 'PMax MW')

        # line flow limit dict
        self.line_flow_limit_dict = df_col_to_dict(self.branch_df,'UID','Cont Rating')

        # number of thermal units
        self.bus_unit_num_dict = {bus: len(get_data_given(self.thermal_gen_param_df,\
        bus = bus)) for bus in self.bus_id}

        self.bus_renewable_num_dict = {bus: len(get_data_given(\
        self.renewable_gen_param_df,bus = bus)) for bus in self.bus_id}

        # construct the network
        #self.G = self.construct_network()

    def print_summary(self):

        '''
        This function prints the summary information on RTS-GMLC dataset, e.g.
        the total number of buses.

        Arguments:
            self: class instance itself.
        Return:
            None
        '''

        print('The number of buses is ', self.num_buses)
        print('The number of thermal generators is ', self.num_thermal_generators)
        print('The number of renewable generators is ', self.num_renewable_generators)
        print('The total number of generators is ', self.total_num_generator)
        print('The amount of thermal power is {} MW'.format(self.total_thermal_power))
        print('The amount of renewable power is {} MW'.format(self.total_renewable_power))
        print('The total amount of power is {} MW'.format(self.total_power))

        return

class PrescientSimulationData(NetworkData):

    ''' Add methods to this class to visualize Prescient results for a single
    generator. These visualizations will NOT depend on the network structure,
    but generator parameters, such as Pmax are pulled from RTS-GMLC dataset.
    '''

    HighPrice = 100

    def __init__(self,result_data_dir,network_data_dir = '../RTS-GMLC/RTS_Data/SourceData/',\
                 custom_string = 'bidding=False_',custom_string2 = 'track_sced_'):
        '''
        This class reads in Prescient simulation results, e.g. thermal_detail.csv,
        and using the data to make plots. This class is a subclass of NetworkData,
        because some data, e.g. the nameplate capacity, is only available in
        that dataset.

        Arguments:
            result_data_dir: directory containing results from Prescient simulation [string]
            network_data_dir: directory containing RTS-GMLC raw data [string]
            custom_string: string added to middle of Prescient results file names [string]

        '''

        # inherit from network data class
        super().__init__(network_data_dir)

        assert isinstance(result_data_dir, str),\
         "Arugment result_data_dir is the wrong type! It should be a string."
        assert isinstance(custom_string, str),\
         "Arugment custom_string is the wrong type! It should be a string."

        self.result_data_dir = result_data_dir
        self.custom_string = custom_string
        self.custom_string2 = custom_string2

        self.read_result_files()

    def read_result_files(self):
        '''
        This function reads full Prescient result files and set them as class
        attributes.
        '''

        # bus details (this has LMP)
        self.bus_detail_df = pd.read_csv(self.result_data_dir + \
                                         self.custom_string+'bus_detail.csv')

        # thermal detail (this has the power delivered from each generator)
        self.thermal_detail_df = pd.read_csv(self.result_data_dir + \
                                             self.custom_string + \
                                             self.custom_string2 + \
                                             'thermal_detail.csv')

        # renewable details
        self.renewable_detail_df = pd.read_csv(self.result_data_dir + \
                                               'renewables_detail.csv')

        # line detail (this has the power flow on each line)
        self.line_detail_df = pd.read_csv(self.result_data_dir + \
                                          'line_detail.csv')

        # hourly summary
        self.hourly_summary_df = pd.read_csv(self.result_data_dir + \
                                             'hourly_summary.csv')

        # the list of unique thermal generators
        self.generator_list = pd.unique(self.thermal_detail_df['Generator'])

        # the list of unique renewable power plants
        self.renewable_list = pd.unique(self.renewable_detail_df['Generator'])

        return

    def summarize_results(self,result_num_decimals = 4,include_generator_param = False,cap_lmp = None):
        '''
        This function summarizes the results for each generator into a dataframe.
        The summary information includes:
            Energy Delivered: MWh
            Energy Averaged Price: $/MWh
            Time On: Hours
            Time On: Fraction
            Capacity Factor: MWh (delivered) / MWh (if at 100% all time)
            Startup: Number of start-up events
            Shutdown: Number of shhut-down events
            Average Time On: hours
            Average Time Off: hours
            Total Uplift Payments: $
            Number of Uplift Days
            Scaled Mileage: sum |Power(t=i+1) - Power(t=i) | / Name Plate
            Generator Characteristics

        Arguments:
            include_generator_param: if True, add generator characteristics from
            RTS-GMLC dataset to the summary dataframe [bool]
        Returns:
            df: the summary dataframe

        '''

        total_dispatch = []
        capacity_factor = []
        total_online_hour = []
        online_fraction = []
        total_offline_hour = []
        offline_fraction = []
        total_start_up = []
        start_up_fraction = []
        shut_down_fraction = []
        total_shut_down = []
        total_uplift_payment = []
        total_uplift_days = []
        total_mileage = []
        average_time_on = []
        average_time_off = []
        average_price = []
        total_revenue = []
        total_cost = []
        total_profit = []
        num_high_price_day = []

        # how many hours of results we have?
        total_result_horizon = len(self.hourly_summary_df)

        # loop thru all the generators
        for generator in self.generator_list:

            # get pmax
            pmax = float(get_data_given(self.gen_param_df,generator = generator)['PMax MW'])

            # get bus
            bus = int(get_data_given(self.gen_param_df,generator = generator)['Bus ID'])
            bus_name = self.bus_id_to_bus_name[bus]

            # total dispatch
            dispatch = get_data_given(self.thermal_detail_df, \
                                      generator = generator)['Dispatch']
            dispatch_da = get_data_given(self.thermal_detail_df, \
                                      generator = generator)['Dispatch DA']
            total_dispatch.append(dispatch.sum())
            capacity_factor.append(total_dispatch[-1]/(pmax*total_result_horizon) * 100)
            dispatch_arr = np.insert(dispatch.values,0,0)
            dispatch_diff = np.diff(dispatch_arr)
            total_mileage.append(np.absolute(dispatch_diff).sum()/(pmax*total_result_horizon) * 100)

            # total on hours
            unit_state = get_data_given(self.thermal_detail_df, \
                                        generator = generator)['Unit State']
            total_online_hour.append(unit_state.sum())

            online_fraction.append(total_online_hour[-1]/total_result_horizon * 100)

            total_offline_hour.append(total_result_horizon - total_online_hour[-1])
            offline_fraction.append(100 - online_fraction[-1])

            # find start up and shut down
            unit_state = np.insert(unit_state.values,0,0) # assume generator is off before the horizon
            unit_state_diff = np.diff(unit_state)
            total_start_up.append(len(np.where(unit_state_diff == 1)[0]))
            total_shut_down.append(len(np.where(unit_state_diff == -1)[0]))
            start_up_fraction.append(total_start_up[-1]/total_result_horizon*100)
            shut_down_fraction.append(total_shut_down[-1]/total_result_horizon*100)

            # find on time and off time
            unit_state = get_data_given(self.thermal_detail_df, \
                                        generator = generator)['Unit State'].values
            online_time, offline_time = self.calc_on_off_hours(unit_state)
            average_time_on.append(np.array(online_time).mean())
            average_time_off.append(np.array(offline_time).mean())

            uplfit_payment = get_data_given(self.thermal_detail_df, \
                                            generator = generator)['Unit Uplift Payment']
            total_uplift_payment.append(uplfit_payment.values.sum())
            total_uplift_days.append(len(np.where(uplfit_payment.values>0)[0]))

            cost = get_data_given(self.thermal_detail_df, \
                                  generator = generator)['Unit Cost']
            total_cost.append(cost.sum())

            lmp = self.bus_detail_df.loc[self.bus_detail_df['Bus'] == bus_name,'LMP']
            lmp_da = self.bus_detail_df.loc[self.bus_detail_df['Bus'] == bus_name,'LMP DA']

            # cooper plate
            # becaus under cooper plate mode all buses names become 'CopperSheet'
            # there is 0 bus matches the original bus name, thus len == 0
            if len(lmp) == 0:
                lmp = self.bus_detail_df.loc[self.bus_detail_df['Bus'] == 'CopperSheet','LMP']
            if len(lmp_da) == 0:
                lmp_da = self.bus_detail_df.loc[self.bus_detail_df['Bus'] == 'CopperSheet','LMP DA']

            #Cap LMP
            if cap_lmp != None:
                lmp[lmp > 200] = cap_lmp

            # count number of days with 'high price'
            num_high_price_day.append(len(np.where(lmp.values>self.HighPrice)[0]))

            # Because sometimes we have NaN in dispatch_da, lmp_da,
            # uplfit_payment but we do not want revenue to be NaN,
            # I need to take adavantage of np.nansum and np.nanprod to calculate
            # the following equation:
            # revenue = lmp.values * (dispatch.values - dispatch_da.values)\
            #           + dispatch_da.values * lmp_da.values\
            #           + uplfit_payment.values

            # difference in dispatches from DAM and RTM
            dispatch_diff = np.nansum(np.vstack((dispatch.values,-dispatch_da.values)),axis = 0)

            # revenue from RTM (first line in the formula)
            rtm_revenue = np.nanprod(np.vstack((lmp.values,dispatch_diff)),axis = 0)

            # revenue from DAM (second line in the formula)
            dam_revenue = np.nanprod(np.vstack((lmp_da.values,dispatch_da.values)),axis = 0)

            # revenue (the whole formula)
            revenue = np.nansum(np.vstack((rtm_revenue,dam_revenue,uplfit_payment.values)),axis = 0)

            total_revenue.append(revenue.sum())
            total_profit.append(total_revenue[-1] - total_cost[-1])
            average_price.append(total_revenue[-1]/total_dispatch[-1])

        # assemble a dataframe
        df = pd.DataFrame(list(zip(self.generator_list,\
        total_dispatch,\
        capacity_factor,\
        total_mileage,\
        total_online_hour,\
        online_fraction,\
        total_offline_hour,\
        offline_fraction,\
        average_time_on,\
        average_time_off,\
        total_start_up,\
        start_up_fraction,\
        total_shut_down,\
        shut_down_fraction,\
        total_uplift_payment,\
        total_uplift_days,\
        average_price,\
        total_cost,\
        total_revenue,\
        total_profit,\
        num_high_price_day)),\
        columns = ['GEN UID',\
        'Total Dispatch [MW]',\
        'Capacity Factor [%]',\
        'Scaled Total Mileage [%]',\
        'Total Online Time [hr]',\
        'Online Fraction [%]',\
        'Total Offline Time [hr]',\
        'Offline Fraction [%]',\
        'Average Online Time [hr]',\
        'Average Offline Time [hr]',\
        'Total Start-up',\
        'Start-up Fraction [%]',\
        'Total Shut-down',\
        'Shut-down Fraction [%]',\
        'Total Uplift Payment [$]',\
        'Total Uplift Payment Days',\
        'Average LMP [$/MWh]',\
        'Total Cost [$]',\
        'Total Revenue [$]',\
        'Total Profit [$]',\
        'Number of Days with High Price'])

        if include_generator_param:
            df = pd.merge(self.thermal_gen_param_df,df,on = ['GEN UID'])

        return df.round({'Total Dispatch [MW]': result_num_decimals,\
        'Capacity Factor [%]': result_num_decimals,\
        'Scaled Total Mileage [%]': result_num_decimals,\
        'Online Fraction [%]': result_num_decimals,\
        'Offline Fraction [%]': result_num_decimals,\
        'Start-up Fraction [%]': result_num_decimals,\
        'Shut-down Fraction [%]': result_num_decimals,\
        'Total Uplift Payment [$]': result_num_decimals,\
        'Average LMP [$/MWh]': result_num_decimals,\
        'Total Cost [$]': result_num_decimals,\
        'Total Revenue [$]': result_num_decimals,\
        'Total Profit [$]': result_num_decimals,\
        'Average Online Time [hr]':result_num_decimals,\
        'Average Offline Time [hr]':result_num_decimals})

    @staticmethod
    def calc_on_off_hours(unit_state):

        '''
        This function calculates the lengths of online and offline periods given
        a list of unit states.

        Arguments:
            unit_state: a list/array of generator on/off state. 1 is on and 0 is
            off.
        Return:
            online_time: a list containing the lengths of each online periods
            offline_time: a list containing the lengths of each offline periods

        Example:
            Given a list of states: unit_state = [1,1,1,0,0,0,1,1].
            It will return:
            online_time = [3,2]
            offline_time = [3]
        '''

        # initialize the lower index of the list
        lo = 0

        # initialize 2 lists to store online time and offline time
        online_time = []
        offline_time = []

        for idx in range(1,len(unit_state)):

            if (unit_state[idx] != unit_state[idx-1]) and (idx != len(unit_state) -1):

                # record offline time
                if unit_state[idx] == 1:
                    offline_time.append(idx - lo)

                # record online time
                else:
                    online_time.append(idx - lo)

                # move the lower pointer to the current idx
                lo = idx

            # this if statement is necessary to handle the last time step
            # imagine a case where no start up occurs
            elif idx == len(unit_state) -1:

                if unit_state[idx] == 1:
                    online_time.append(idx - lo + 1)
                else:
                    offline_time.append(idx - lo + 1)

        # post processing: if the list is empty, it means no event happened, so
        # 0 online time or 0 offline time
        if len(online_time) == 0:
            online_time.append(0)
        if len(offline_time) == 0:
            offline_time.append(0)

        return online_time, offline_time

################################################################################
############################ Other Plotting functions ##########################
################################################################################

    def price_histogram_given_time(self,date,hour):

        '''
        plot the histogram of lmps given date and hour across all the nodes.

        Arguments:
            date: the date we are interested in. [str]
            hour: the hour we are interested in. [int]
        '''

        bus_time_detail_df = get_data_given(self.bus_detail_df,date = date,hour = hour)
        lmp = bus_time_detail_df['LMP']
        ax = lmp.hist(bins = 10)

        ax.set_xlabel('LMP [$/MWh]',fontsize = 15)
        ax.set_ylabel('Frequency',fontsize = 15)
        ax.tick_params(labelsize= 15)
        ax.set_title('{} Hour = {} LMP Histogram'.format(date,hour),fontsize = 15)

        return ax


class ExtractedPrescientSimulationData():

    ''' Add methods to this class to visualize Prescient results for a single
    generator. These visualizations will NOT depend on the network structure,
    but generator parameters, such as Pmax are pulled from RTS-GMLC dataset.
    '''

    summary_col = [\
    'Total Dispatch [MW]',\
    'Capacity Factor [%]',\
    'Scaled Total Mileage [%]',\
    'Total Online Time [hr]',\
    'Online Fraction [%]',\
    'Total Offline Time [hr]',\
    'Offline Fraction [%]',\
    'Average Online Time [hr]',\
    'Average Offline Time [hr]',\
    'Total Start-up',\
    'Start-up Fraction [%]',\
    'Total Shut-down',\
    'Shut-down Fraction [%]',\
    'Average LMP [$/MWh]',\
    'Total Cost [$]',\
    'Total Revenue [$]',\
    'Total Profit [$]']

    gen_param_col = [\
    'PMax [MW]',\
    'PMin [MW]',\
    'Ramp Rate [MW/hr]',\
    'Min Up Time [Hr]',\
    'Min Down Time [Hr]',\
    'Marginal Cost [$/MWh]',\
    'No Load Cost [$/hr]',\
    'Start Time Hot [Hr]',\
    'Start Time Warm [Hr]',\
    'Start Time Cold [Hr]',\
    'Start Cost Hot [$]',\
    'Start Cost Warm [$]',\
    'Start Cost Cold [$]']

    def __init__(self,result_data_dir,param_data_dir):
        '''
        This class reads in extracted Prescient simulation results from SNL, and
        summarize the results.

        Arguments:
            result_data_dir: directory containing results from Prescient
            simulation [string]
        '''

        assert isinstance(result_data_dir, str),\
         "Arugment result_data_dir is the wrong type! It should be a string."

        self.result_data_dir = result_data_dir
        self.param_data_dir = param_data_dir
        self.read_result_files()

    def read_result_files(self):
        '''
        This function reads full Prescient result files and set them as class
        attributes.
        '''

        # thermal detail (this has the power delivered from each generator)
        self.thermal_detail_df = pd.read_csv(self.result_data_dir)

        # read the params perturbed
        with open(self.param_data_dir,'r') as f:
            self.param_data = json.load(f)

        return

    def summarize_results(self,result_num_decimals = 4,return_numpy_arr = False,\
                          include_generator_param = False, cap_lmp=False):
        '''
        This function summarizes the results for each generator into a dataframe.
        The summary information includes:
            Energy Delivered: MWh
            Energy Averaged Price: $/MWh
            Time On: Hours
            Time On: Fraction
            Capacity Factor: MWh (delivered) / MWh (if at 100% all time)
            Startup: Number of start-up events
            Shutdown: Number of shhut-down events
            Average Time On: hours
            Average Time Off: hours
            Total Uplift Payments: $
            Number of Uplift Days
            Scaled Mileage: sum |Power(t=i+1) - Power(t=i) | / Name Plate
            Generator Characteristics

        Arguments:
            result_num_decimals: number of decimal in the result
            return_numpy_arr: if True, return the summary info as a numpy 1D
            array; Otherwise, return it as a pandas dataframe.
            include_generator_param:
            include_generator_param: if True, add generator characteristics from
            the json file [bool]
        Returns:
            if return_numpy_arr is true:
                the summay info as a numpy array
            else:
                df: the summary dataframe
        '''

        # how many hours of results we have?
        total_result_horizon = len(self.thermal_detail_df)

        # get pmax
        pmax = self.param_data['p_max']

        # total dispatch
        dispatch = self.thermal_detail_df['Dispatch']
        total_dispatch = dispatch.sum()
        capacity_factor = (total_dispatch/(pmax*total_result_horizon) * 100)
        dispatch_arr = np.insert(dispatch.values,0,0)
        dispatch_diff = np.diff(dispatch_arr)
        total_mileage = (np.absolute(dispatch_diff).sum()/(pmax*total_result_horizon) * 100)

        # total on hours
        unit_state = self.thermal_detail_df['Unit State']
        total_online_hour = unit_state.sum()

        online_fraction = total_online_hour/total_result_horizon * 100

        total_offline_hour = (total_result_horizon - total_online_hour)
        offline_fraction = (100 - online_fraction)

        # find start up and shut down
        unit_state = np.insert(unit_state.values,0,0) # assume generator is off before the horizon
        unit_state_diff = np.diff(unit_state)
        total_start_up = len(np.where(unit_state_diff == 1)[0])
        total_shut_down = len(np.where(unit_state_diff == -1)[0])
        start_up_fraction = total_start_up/total_result_horizon*100
        shut_down_fraction = total_shut_down/total_result_horizon*100

        # find on time and off time
        unit_state = self.thermal_detail_df['Unit State']
        online_time, offline_time = PrescientSimulationData.calc_on_off_hours(unit_state.values)
        average_time_on = np.array(online_time).mean()
        average_time_off = np.array(offline_time).mean()

        # calculate the total cost using the params and result data
        production_cost = dispatch.values * self.param_data['marginal_cost'] +\
                          unit_state.values * self.param_data['no_load_cost']
        startup_cost = self.calc_startup_cost()

        total_cost = production_cost.sum() + startup_cost.sum()

        lmp = self.thermal_detail_df['LMP']
        if cap_lmp:
            lmp[lmp>cap_lmp] = cap_lmp

        revenue = lmp.values * dispatch.values
        total_revenue = revenue.sum()
        total_profit = total_revenue - total_cost
        average_price = total_revenue/total_dispatch

        # assemble a dataframe
        df = pd.DataFrame([\
        total_dispatch,\
        capacity_factor,\
        total_mileage,\
        total_online_hour,\
        online_fraction,\
        total_offline_hour,\
        offline_fraction,\
        average_time_on,\
        average_time_off,\
        total_start_up,\
        start_up_fraction,\
        total_shut_down,\
        shut_down_fraction,\
        average_price,\
        total_cost,\
        total_revenue,\
        total_profit],\
        index = self.summary_col).T

        if include_generator_param:
            gen_param_df = self.gen_param_to_df()
            df  = pd.concat([gen_param_df,df],sort = False,axis = 1)

        df = df.round({'Total Dispatch [MW]': result_num_decimals,\
        'Capacity Factor [%]': result_num_decimals,\
        'Scaled Total Mileage [%]': result_num_decimals,\
        'Online Fraction [%]': result_num_decimals,\
        'Offline Fraction [%]': result_num_decimals,\
        'Start-up Fraction [%]': result_num_decimals,\
        'Shut-down Fraction [%]': result_num_decimals,\
        'Average LMP [$/MWh]': result_num_decimals,\
        'Total Cost [$]': result_num_decimals,\
        'Total Revenue [$]': result_num_decimals,\
        'Total Profit [$]': result_num_decimals,\
        'Average Online Time [hr]': result_num_decimals,\
        'Average Offline Time [hr]': result_num_decimals,\
        'PMax [MW]': result_num_decimals,\
        'PMin [MW]': result_num_decimals,\
        'Ramp Rate [MW/hr]': result_num_decimals,\
        'Min Up Time [Hr]': result_num_decimals,\
        'Min Down Time [Hr]': result_num_decimals,\
        'Marginal Cost [$/MWh]': result_num_decimals,\
        'No Load Cost [$/hr]': result_num_decimals,\
        'Start Time Hot [Hr]': result_num_decimals,\
        'Start Time Warm [Hr]': result_num_decimals,\
        'Start Time Cold [Hr]': result_num_decimals,\
        'Start Cost Hot [$]': result_num_decimals,\
        'Start Cost Warm [$]': result_num_decimals,\
        'Start Cost Cold [$]': result_num_decimals})

        if return_numpy_arr:
            return df.values.flatten()
        else:
            return df

    # a method to calculate start up cost
    def calc_startup_cost(self):

        '''
        This function calculate the start up cost using the param_data json file
        and the unit state.

        Arguments:
            self
        Returns:
            startup_cost_arr: a 1D numpy array of start up costs. Each element is
            corresponding to each cost of each start up. It does not have the
            length of simulation horizon.
        '''

        # get the unit state
        unit_state = self.thermal_detail_df['Unit State']

        # assume all the generators have been off for 1 hour before start of horizon
        # so I insert one 0 at the beginning of the unit state
        unit_state = np.insert(unit_state.values,0,0)

        # calc down time
        online_time_list, offline_time_list = PrescientSimulationData.calc_on_off_hours(unit_state)

        # get startup cost profile
        # the first col is [hot start time, warm start time, cold start time]
        # the second col is the corresponding costs
        startup_cost_profile = np.array(self.param_data['startup_cost_profile'])

        startup_cost_list = []
        for offline_time in offline_time_list:

            # find out whether it is a hot, warm, or cold start
            cost_category_idx = np.searchsorted(startup_cost_profile[:,0],offline_time)

            # cold start: down time could be arbitary long, so it can be larger
            # than the listed cold start time and the idx is == to the length
            if cost_category_idx >= len(startup_cost_profile):
                cost_category_idx = len(startup_cost_profile) - 1

            startup_cost_list.append(startup_cost_profile[cost_category_idx,1])

        startup_cost_arr = np.array(startup_cost_list)

        return startup_cost_arr

    def gen_param_to_df(self):

        pmax = self.param_data['p_max']
        pmin = pmax * self.param_data['p_min_multi']
        ramp_rate = (pmax - pmin) * self.param_data['ramp_multi']

        min_up_time = self.param_data['min_up']
        min_dn_time = min_up_time * self.param_data['min_dn_multi']

        marginal_cost = self.param_data['marginal_cost']
        no_load_cost = self.param_data['no_load_cost']

        # make sure there are 3 pairs of start up costs corresponding to hot,
        # warm and cold starts
        while len(self.param_data['startup_cost_profile']) < 3:
            self.param_data['startup_cost_profile'].append(self.param_data['startup_cost_profile'][-1])

        hot_start_time = self.param_data['startup_cost_profile'][0][0]
        warm_start_time = self.param_data['startup_cost_profile'][1][0]
        cold_start_time = self.param_data['startup_cost_profile'][2][0]

        hot_start_cost = self.param_data['startup_cost_profile'][0][1]
        warm_start_cost = self.param_data['startup_cost_profile'][1][1]
        cold_start_cost = self.param_data['startup_cost_profile'][2][1]

        # assemble a dataframe
        df = pd.DataFrame([\
        pmax,\
        pmin,\
        ramp_rate,\
        min_up_time,\
        min_dn_time,\
        marginal_cost,\
        no_load_cost,\
        hot_start_time,\
        warm_start_time,\
        cold_start_time,\
        hot_start_cost,\
        warm_start_cost,\
        cold_start_cost,\
        ],\
        index = self.gen_param_col).T

        return df

    def calc_dispatch_zones(self):
        thermal_data = sim.thermal_detail_df

