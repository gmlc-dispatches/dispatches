#This script plots the surrogate results for a single rankine cycle solution. It creates plots for:
#1.) Operating Profile
# It also prints out revenue, capex, opex, and net revenue over 20 years
import json
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)

#json contains surrogate solution
# with open("rankine_nn_p_max_lower_175.json") as f:
#     data = json.load(f)
with open("./results_solutions_neuralnetwork/rankine_nn_175_fix_startup_profile_blue.json") as f:
    data = json.load(f)

## THE CONSTANTS FOR THIS RUN
x = data["market_inputs"]
pmax = x[0]
pmin = x[1]
ramp_rate = x[2]
min_up_time = x[3]
min_down_time = x[4]
marginal_cost = x[5]
fixed_run_cost = x[6]
st_time_hot = x[7]
st_time_warm = x[8]
st_time_cold = x[9]
st_cst_hot = x[10]
st_cst_warm = x[11]
st_cst_cold = x[12]

startup_cost_profile = [(st_time_hot,st_cst_hot),(st_time_warm,st_cst_warm),(st_time_cold,st_cst_cold)]

dispatch_zones = data["scaled_dispatch_zones"]

fig, ax = plt.subplots(figsize = (16,8))
ax.set_xlabel("Scaled Power Output (% of maximum)", fontsize=24)
ax.set_xticks(range(len(dispatch_zones)))
ax.tick_params(axis='x', labelrotation = 45)
ax.set_xticklabels(["Off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"])

ax.bar(range(len(dispatch_zones)),dispatch_zones, color="blue")
ax.set_ylabel("Hours in Operating Zone", fontsize=24)
plt.tight_layout()
fig.savefig("results_solutions_neuralnetwork/zone_operation_surrogate_yellow_175.png")
