#This is a plugin to run the surrogate model inputs through Prescient for verification
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from driver_verification import base_output_dir
import math
import json, os

#json contains surrogate solution
design_solution_path = os.path.join(this_file_dir(),"../surrogate_design/rankine_cycle_case/\
design_results/scikit_surrogate/conceptual_design_solution_nn_0.json")

design_solution_path = os.path.join(this_file_dir(),"../surrogate_design/rankine_cycle_case/\
design_results/scikit_surrogate/conceptual_design_solution_nn_1.json")

design_solution_path = os.path.join(this_file_dir(),"../surrogate_design/rankine_cycle_case/\
design_results/scikit_surrogate/conceptual_design_solution_nn_2.json")

design_solution_path = os.path.join(this_file_dir(),"../surrogate_design/rankine_cycle_case/\
design_results/alamo_surrogate/conceptual_design_solution_alamo_0.json")

with open(os.path.join(this_file_dir()+"/../rankine_results/{}".format(design_solution_filename))) as f:
    data = json.load(f)

#values are: [lag,cost], units are: [hr/min_dn (hr),$/MW capacity]
startup_cost_data = { 'yellow' : [ (0.75, 94.00023429), (2.5, 135.2230393), (3, 147.0001888) ],
                      'blue'   : [ (0.375, 93.99890632), (1.375, 101.4374234), (7.5, 146.9986814) ],
                      'brown'  : [ (0.166666667, 58.99964891), (0.25, 61.09068702), (2, 104.9994673) ],
                      'dark_blue': [ (0.111111111, 35.00249986), (0.222222222, 49.66991167), (0.444444444, 79.00473527) ],
                     }

#make profile 0 no cost, and profile 4 the highest cost
startup_cost_profiles = [   [ (1.0, 0.) ],
                            startup_cost_data['dark_blue'],
                            startup_cost_data['brown'],
                            startup_cost_data['blue'],
                            startup_cost_data['yellow']]

## THE CONSTANTS FOR THIS RUN
x = data["market_inputs"]
pmax = x[0]
p_min_multi = x[1]
ramp_multi = x[2]
min_up_time = int(math.ceil(x[3]))
min_dn_multi = int(math.ceil(x[4]))
marginal_cost = x[5]
fixed_run_cost = x[6]
startup_index = 2
startup_cost_profile = startup_cost_profiles[startup_index]

parameters = {'pmax':pmax, 'p_min_multi':p_min_multi, 'ramp_multi':ramp_multi, 'min_up_time':min_up_time,
'min_dn_multi':min_dn_multi, 'marginal_cost':marginal_cost, 'fixed_run_cost':fixed_run_cost, 'startup_profile':startup_index}
with open(base_output_dir+'/parameters.json', 'w') as parmfile:
    json.dump(parameters,parmfile)

pmin = p_min_multi*pmax
ramp_rate = ramp_multi*(pmax-pmin)
min_down_time = int(math.ceil(min_dn_multi*min_up_time))

gen = '123_STEAM_3'
def change_gen_123_STEAM_3(data, market):

    hr_ramp_rate = ramp_rate
    startup_shutdown_rate = min(pmin+0.5*hr_ramp_rate, pmax)

    min_up = min_up_time
    min_dn = min_down_time

    #Get data dictionary
    data_none = data[None]
    ## change the p_max
    data_none['MaximumPowerOutput'][gen] = pmax

    ## change the p_min
    data_none['MinimumPowerOutput'][gen] = pmin

    ## fix the initial state, if needed (for the initial inital conditions)
    power_gen_t0 = pyo.value(data_none['PowerGeneratedT0'][gen])
    unit_on_t0 = pyo.value(data_none['UnitOnT0'][gen])

    if unit_on_t0:
        if power_gen_t0 > pmax:
            data_none['PowerGeneratedT0'][gen] = pmax
        if power_gen_t0 < pmin:
            data_none['PowerGeneratedT0'][gen] = pmin

    ## change the ramp rate
    data_none['NominalRampUpLimit'][gen] = hr_ramp_rate
    data_none['NominalRampDownLimit'][gen] = hr_ramp_rate

    ## change the startup/shutdown ramp rate
    data_none['StartupRampLimit'][gen] = startup_shutdown_rate
    data_none['ShutdownRampLimit'][gen] = startup_shutdown_rate

    ## change the cost
    data_none['CostPiecewisePoints'][gen] = [pmin, pmax]
    data_none['CostPiecewiseValues'][gen] = [pmax*fixed_run_cost + pmin*marginal_cost,
                                            pmax*fixed_run_cost + pmax*marginal_cost]

    ## change the uptime/downtime
    data_none['MinimumUpTime'][gen] = min_up
    data_none['MinimumDownTime'][gen] = min_dn

    raw_startup_costs = [ [min_dn*time_over_min_dn, pmax*cost_over_capacity] \
                        for time_over_min_dn, cost_over_capacity in startup_cost_profile ]

    for idx, (startup_time, _) in enumerate(raw_startup_costs):
        if startup_time > min_dn:
            ## go back to the index before this was the case
            idx -= 1
            break
    startup_costs = raw_startup_costs[idx:]

    startup_costs[0][0] = min_dn

    startup_costs_rounded = [ (int(math.ceil(lag)), cost) for lag, cost in startup_costs ]

    startup_costs = []
    ## eliminate duplicates from math.ceil
    for idx, (lag, cost) in enumerate(startup_costs_rounded[:-1]):
        next_lag = startup_costs_rounded[idx+1][0]
        if lag == next_lag:
            continue
        else:
            startup_costs.append((lag,cost))
    ## put on the last cost
    startup_costs.append(startup_costs_rounded[-1])

    data_none['StartupLags'][gen] = [ lag for lag, _ in startup_costs ]
    data_none['StartupCosts'][gen] = [ cost for _, cost in startup_costs ]

    print("MODIFIED 123 STEAM 3")

data_dict_callback = change_gen_123_STEAM_3

print("PARSED VERIFICATION PLUGIN")
