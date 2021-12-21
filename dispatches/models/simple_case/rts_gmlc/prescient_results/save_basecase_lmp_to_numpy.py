import pandas as pd
import numpy as np
import os

result_data_dir = 'basecase_prescient_runs/deterministic_simulation_basecase'
network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/'

bus_detail_df = pd.read_csv(os.path.join(result_data_dir,'bus_detail.csv'))

# thermal detail (power dispatched by each generator)
thermal_detail_df = pd.read_csv(os.path.join(result_data_dir,'thermal_detail.csv'))

# renewable details
renewable_detail_df = pd.read_csv(os.path.join(result_data_dir,'renewables_detail.csv'))

# line detail (this has the power flow on each line)
line_detail_df = pd.read_csv(os.path.join(result_data_dir,'line_detail.csv'))

# hourly summary
hourly_summary_df = pd.read_csv(os.path.join(result_data_dir,'hourly_summary.csv'))

# the list of unique thermal generators
generator_list = pd.unique(thermal_detail_df['Generator'])

# the list of unique renewable power plants
renewable_list = pd.unique(renewable_detail_df['Generator'])

#Look at an example coal generator (This is the generator we studied for the big Prescient simulation data set)
coal_generator = '123_STEAM_3'
gen_results = thermal_detail_df.loc[thermal_detail_df['Generator'] == coal_generator] #output results for this generator
dispatch = gen_results["Dispatch"].to_numpy()

bus_results = bus_detail_df.loc[bus_detail_df['Bus'] == "CopperSheet"] #output results for this generator
lmp = np.copy(bus_results["LMP"])
lmp[lmp > 200] = 200

#Save prices and dispatch
with open('rts_results_all_prices_base_case.npy', 'wb') as f:
    np.save(f,dispatch)
    np.save(f,lmp)


##############################################
#RTS-GMLC network input data
#This data has network parameters such as bus and generator names. It is needed to get prices for
#specific buses if not using the coppersheet assumption
##############################################
network_data_dir = os.path.abspath('../../RTS-GMLC/RTS_Data/SourceData')

bus_df = pd.read_csv(os.path.join(network_data_dir,'bus.csv'))
branch_df = pd.read_csv(os.path.join(network_data_dir,'branch.csv'))

# generator params (this has the capacity of each generator)
gen_param_df = pd.read_csv(os.path.join(network_data_dir,'gen.csv'))

# thermal generators df
dispatchable_fuel_types = ['Coal','Oil','NG','Nuclear']
thermal_gen_param_df = gen_param_df.loc[gen_param_df['Fuel'].isin(dispatchable_fuel_types)]

# renewable generators df
renewable_fuel_types = ['Hydro','Solar','Wind']
renewable_gen_param_df = gen_param_df.loc[gen_param_df['Fuel'].isin(renewable_fuel_types)]

#Look at coal generator parameters
coal_gen_data = gen_param_df.loc[gen_param_df['GEN UID'] == coal_generator] #input generator parameters to simulation

bus_id = np.array(coal_gen_data["Bus ID"])[0]
bus_name = np.array(bus_df.loc[bus_df['Bus ID'] == bus_id]["Bus Name"])[0]
bus_results = bus_detail_df.loc[bus_detail_df['Bus'] == bus_name]
lmp = np.copy(bus_results["LMP"])
lmp[lmp > 200] = 200
