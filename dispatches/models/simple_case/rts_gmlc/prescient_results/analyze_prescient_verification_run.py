import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=18)
plt.rc('axes', titlesize=18)
import os
from prescient_analysis_code import PrescientSimulationData
import json

# result_data_dir = 'prescient_verification_results/pmin_175_nn_case_1/'
##############################################
#Prescient Simulation Results
##############################################
# result_data_dir = 'prescient_verification_results/pmin_175_nn_blue_startup/'
# network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/'

# #Load up the json file to read parameters
# with open(result_data_dir+"rankine_nn_175_fix_startup_profile_blue.json") as f:
#     surrogate_data = json.load(f)

# result_data_dir = 'prescient_verification_results/pmin_175_nn_free_startup_2/'
result_data_dir = 'prescient_verification_results/pmin_175_nn_blue_startup/'
network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/'

#Load up the json file to read parameters
with open(result_data_dir+"rankine_nn_175_fix_startup_profile_blue.json") as f:
    surrogate_data = json.load(f)

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

##############################################
#Obtained input data using RTS-GMLC network
##############################################
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
lmp_da = np.copy(bus_results["LMP DA"])

#Simulation result for Coal Gen 123-STEAM-3
coal_generator = '123_STEAM_3'
gen_results = thermal_detail_df.loc[thermal_detail_df['Generator'] == coal_generator] #output results for this generator
dispatch = gen_results["Dispatch"]

##############################################
# Calculate total revenue
##############################################
#more complex revenue, not really captures in Prescient runs
dispatch_da = gen_results["Dispatch DA"]
uplift_payment = gen_results['Unit Uplift Payment']
dispatch_diff = np.nansum(np.vstack((dispatch.values,-dispatch_da.values)),axis = 0)
rtm_revenue = np.nanprod(np.vstack((lmp,dispatch_diff)),axis = 0)
dam_revenue = np.nanprod(np.vstack((lmp_da,dispatch_da.values)),axis = 0)
revenue = np.nansum(np.vstack((rtm_revenue,dam_revenue,uplift_payment.values)),axis = 0)
total_revenue = sum(revenue)/1e6

#simple revenue calculation using only prices and dispatch in rtm
total_dispatch = dispatch.sum()
total_revenue_2 = sum(dispatch_diff*lmp)

##############################################
# Calculate dispatch zones
##############################################
pmax = surrogate_data["market_inputs"][0]
pmin = surrogate_data["market_inputs"][1]

bins_col_names = ["# Hours 0%","# Hours 0-10%","# Hours 10-20%","# Hours 20-30%","# Hours 30-40%","# Hours 40-50%","# Hours 50-60%","# Hours 60-70%","# Hours 70-80%","# Hours 80-90%","# Hours 90-100%"]
dispatch_ranges = [(0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,70),(70,80),(80,90),(90,100)]

prescient_zone_hours = np.zeros(len(dispatch_ranges)+1)
dispatch_prescient = dispatch.to_numpy()
n_hours_off = len(np.nonzero(dispatch_prescient == 0)[0])

dispatch_on = dispatch_prescient[dispatch_prescient > 0.0]
scaled_dispatch = np.round((dispatch_on - pmin) / (pmax - pmin) * 100,3)

assert(n_hours_off + len(scaled_dispatch) == 8736)
prescient_zone_hours[0] = n_hours_off
for i in range(len(dispatch_ranges)-1):
    zone = dispatch_ranges[i]
    n_hours_zone = len(np.where((scaled_dispatch >= zone[0]) & (scaled_dispatch < zone[1]))[0])
    prescient_zone_hours[i+1] = n_hours_zone
prescient_zone_hours[-1] = len(np.where((scaled_dispatch >= dispatch_ranges[-1][0]) & (scaled_dispatch <= dispatch_ranges[-1][1]))[0])

assert(sum(prescient_zone_hours) == 8736)

##############################################
# Create Plots
##############################################
revenue_surrogate = surrogate_data["revenue_surrogate"]
dispatch_zones = surrogate_data["scaled_dispatch_zones"]

# Compare dispatch zones
# fig, ax = plt.subplots(figsize = (8,8))
# ax.set_xlabel("Scaled Power Output (% of maximum)", fontsize=24)
# ax.set_xticks(range(len(dispatch_zones)))
# ax.tick_params(axis='x', labelrotation = 45)
# ax.set_xticklabels(["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"])
# rects1 = ax.bar(range(len(dispatch_zones)),dispatch_zones, color="blue", label = "Surrogate: Revenue = {}".format(revenue_surrogate))
# rects2 = ax.bar(range(len(dispatch_zones)),prescient_zone_hours, color="green",label = "Prescient: Revenue = {}".format(total_revenue))
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# ax.set_ylabel("Hours in Operating Zone", fontsize=24)
# ax.legend()
# plt.tight_layout()
# fig.savefig(result_data_dir+"zone_operation_surrogate_yellow_175.png")

# Use Dataframe to get the plot
fig, ax = plt.subplots(figsize = (8,8))
ax.set_ylabel("Hours in Operating Zone")
ax.set_xlabel("Scaled Power Output (% of maximum)")
labels = ["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
columns = list(zip(labels,list(dispatch_zones),list(prescient_zone_hours)))
df_plt = pd.DataFrame(columns, columns=["Scaled Power Output (% of maximum)", 
            "NN Surrogate: {} MM Revenue".format(round(revenue_surrogate,1)), 
            "Simulation:     {} MM Revenue".format(round(total_revenue,1))])

df_plt.plot(ax = ax, x="Scaled Power Output (% of maximum)",kind='bar',stacked=False)
ax.tick_params(axis='x', labelrotation = 45)
plt.tight_layout()
plt.savefig(result_data_dir+"zone_operation_surrogate_blue_175.png")

# Plot the dispatch profile
# dispatch_np = dispatch.to_numpy()
# (n, bins, patches) = plt.hist(dispatch_np, bins=100, label='hst')
# # plt.show()

# # LMP Prices
plt.cla()
lmp_prescient = np.array(lmp)
(n, bins, patches) = plt.hist(lmp, bins=100, label='hst')
# plt.show()





