import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import os
#NOTE: This script pull the results from a single Prescient simulation
# import prescient_analyzer as pa

# #result_data_dir = '../deterministic_simulation_output_index_{}/'.format(result_idx)
# result_data_dir = 'simulation_365_days_default'
# network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/'
# sim = pa.PrescientSimulationData(result_data_dir = result_data_dir, network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/', custom_string = '',custom_string2 = '')

# sim.read_result_files()
# # update pmax (some calculations depend on it)
# # pmax = param_data['p_max']
# # sim.gen_param_df.at[sim.gen_param_df['GEN UID'] == generator_perturbed,'PMax MW'] = pmax

# # summary dataframe
# df = sim.summarize_results(include_generator_param = True)

# # write the summary to csv
# df.to_csv(summary_dir + 'result_index_{}.csv'.format(result_idx),index = False)
#generators

#result_data_dir = 'deterministic_simulation_idaes_branch_default'
result_data_dir = 'deterministic_simulation_naerm_branch_basecase'
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
dispatch = gen_results["Dispatch"]


#Plot the dispatch profile
dispatch_np = dispatch.to_numpy()
(n, bins, patches) = plt.hist(dispatch_np, bins=100, label='hst')
plt.show()

##############################################
#Obtained input data using RTS-GMLC network
##############################################
#NOTE: You will need to setup the path to the RTS-GMLC source data
#network_data_dir = os.path.abspath('../../RTS-GMLC/RTS_Data/SourceData')

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
bus_results = bus_detail_df.loc[bus_detail_df['Bus'] == "CopperSheet"] #output results for this generator
lmp = np.copy(bus_results["LMP"])
lmp[lmp > 200] = 200

# lmp_np = lmp.to_numpy()
(n, bins, patches) = plt.hist(lmp_np, bins=100, label='hst')
plt.show()

#BASE CASE DATA
# pmax = 355
# pmin = 0.3*pmax 
# ramp_rate = 0.5*(pmax-pmin)
# min_up_time = 4 
# min_down_time = 1.0*min_up_time
# marginal_cost = 25.0
# fixed_run_cost = 1.0
# startup_cost_profile = startup_cost_profiles[1]


#Save prices and dispatch
# with open('rts_results_all_prices.npy', 'wb') as f:
#     np.save(f,dispatch_np)
#     np.save(f,lmp_np)