import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=18)
plt.rc('axes', titlesize=18)
from pyomo.common.fileutils import this_file_dir
import os
import json

##############################################
#Prescient Simulation Results
##############################################
idx=1
result_data_dir = os.path.join(this_file_dir(),"verification_runs/verification_{}/deterministic_simulation_output_index_{}".format(idx,idx))
surrogate_data_filename = os.path.join(this_file_dir(),"verification_runs/verification_{}/verification_inputs_{}.json".format(idx,idx))

with open(surrogate_data_filename) as f:
    surrogate_data = json.load(f)

bus_detail_df = pd.read_csv(os.path.join(result_data_dir,'bus_detail.csv'))
thermal_detail_df = pd.read_csv(os.path.join(result_data_dir,'thermal_detail.csv'))

#Look at an example coal generator (This is the generator we studied for the big Prescient simulation data set)
coal_generator = '123_STEAM_3'
gen_results = thermal_detail_df.loc[thermal_detail_df['Generator'] == coal_generator] #output results for this generator
dispatch = gen_results["Dispatch"]

# bus_id = np.array(coal_gen_data["Bus ID"])[0]
# bus_name = np.array(bus_df.loc[bus_df['Bus ID'] == bus_id]["Bus Name"])[0]
bus_results = bus_detail_df.loc[bus_detail_df['Bus'] == "CopperSheet"] #output results for this generator
lmp = np.copy(bus_results["LMP"])
lmp[lmp > 200] = 200
lmp_da = np.copy(bus_results["LMP DA"])

##############################################
# Calculate total revenue
##############################################
#more complex revenue, not really captures in Prescient runs
dispatch_da = gen_results["Dispatch DA"]
uplift_payment = gen_results['Unit Uplift Payment']
dispatch_diff = np.nansum(np.vstack((dispatch.values,-dispatch_da.values)),axis = 0)
rtm_revenue = np.nanprod(np.vstack((lmp,dispatch_diff)),axis = 0)
#dam_revenue = np.nanprod(np.vstack((lmp_da,dispatch_da.values)),axis = 0)
#revenue = np.nansum(np.vstack((rtm_revenue,dam_revenue,uplift_payment.values)),axis = 0)
#total_revenue_2 = sum(dispatch_diff*lmp)

total_revenue = sum(rtm_revenue)/1e6
#simple revenue calculation using only prices and dispatch in rtm
total_dispatch = dispatch.sum()


##############################################
# Calculate dispatch zones
##############################################
pmax = surrogate_data["market_inputs"][0]
pmin = surrogate_data["market_inputs"][1]*pmax

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
# Calculate number of startups
##############################################
thermal_detail_diff = gen_results['Unit State'].diff().fillna(0)
state_changes = thermal_detail_diff.value_counts()
if 1 in state_changes.keys():
    n_startups = state_changes[1]
else:
    n_startups = 0 

##############################################
# Create Plots
##############################################
revenue_surrogate = surrogate_data["revenue_surrogate"]
nstartups_surrogate = round(surrogate_data["nstartups_per_year"])
dispatch_zones = surrogate_data["scaled_dispatch_zones"]

# Use Dataframe to get the plot
fig, ax = plt.subplots(figsize = (8,8))
ax.set_ylabel("Hours in Operating Zone")
ax.set_xlabel("Scaled Power Output (% of maximum)")
labels = ["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
columns = list(zip(labels,list(dispatch_zones),list(prescient_zone_hours)))
df_plt = pd.DataFrame(columns, columns=["Scaled Power Output (% of maximum)", 
            "NN Surrogate","Prescient Verification"])

df_plt.plot(ax = ax, x="Scaled Power Output (% of maximum)",kind='bar',stacked=False)
ax.tick_params(axis='x', labelrotation = 45)
plt.tight_layout()
plt.savefig(result_data_dir+"/../zone_verification_{}.png".format(idx))
plt.savefig(result_data_dir+"/../zone_verification_{}.pdf".format(idx))


verification_results = {"revenue_surrogate":revenue_surrogate,"revenue_verify":total_revenue,
                        "nstartups_surrogate":int(nstartups_surrogate),"n_startups_verify":int(n_startups),
                        "dispatch_zones_surrogate":dispatch_zones,"dispatch_zones_verify":list(prescient_zone_hours)}

with open(os.path.join(this_file_dir(),"verification_runs/verification_{}/verification_results_{}.json".format(idx,idx)),"w") as f:
    json.dump(verification_results,f)



# Compare dispatch zones
# fig, ax = plt.subplots(figsize = (8,8))
# ax.set_xlabel("Scaled Power Output (% of maximum)", fontsize=24)
# ax.set_xticks(range(len(dispatch_zones)))
# ax.tick_params(axis='x', labelrotation = 45)
# ax.set_xticklabels(["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"])
# rects1 = ax.bar(range(len(dispatch_zones)),dispatch_zones, color="blue", label = "Surrogate: Revenue = {}".format(round(revenue_surrogate,1))
# rects2 = ax.bar(range(len(dispatch_zones)),prescient_zone_hours, color="green",label = "Prescient: Revenue = {}".format(round(total_revenue,1))
# # ax.bar_label(rects1, padding=3)
# # ax.bar_label(rects2, padding=3)
# ax.set_ylabel("Hours in Operating Zone", fontsize=24)
# ax.legend()
# plt.tight_layout()
# fig.savefig(result_data_dir+"/../zone_verification.png")

# Plot the dispatch profile
# dispatch_np = dispatch.to_numpy()
# (n, bins, patches) = plt.hist(dispatch_np, bins=100, label='hst')
# # plt.show()

# # LMP Prices
# plt.cla()
# lmp_prescient = np.array(lmp)
# (n, bins, patches) = plt.hist(lmp, bins=100, label='hst')
# plt.show()


##############################################
#Obtained input data using RTS-GMLC network
##############################################
#network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/'
# bus_df = pd.read_csv(os.path.join(network_data_dir,'bus.csv'))
# branch_df = pd.read_csv(os.path.join(network_data_dir,'branch.csv'))

# # generator params (this has the capacity of each generator)
# gen_param_df = pd.read_csv(os.path.join(network_data_dir,'gen.csv'))

# # thermal generators df
# dispatchable_fuel_types = ['Coal','Oil','NG','Nuclear']
# thermal_gen_param_df = gen_param_df.loc[gen_param_df['Fuel'].isin(dispatchable_fuel_types)]

# # renewable generators df
# renewable_fuel_types = ['Hydro','Solar','Wind']
# renewable_gen_param_df = gen_param_df.loc[gen_param_df['Fuel'].isin(renewable_fuel_types)]

# #Look at coal generator parameters
# coal_gen_data = gen_param_df.loc[gen_param_df['GEN UID'] == coal_generator] #input generator parameters to simulation



