#This is a plugin to run the surrogate model inputs through Prescient for verification
import pyomo.environ as pyo
import math
import json

#json contains surrogate solution
# with open("rankine_nn_p_max_lower_175.json") as f:
#     data = json.load(f)
with open("../results_solutions_neuralnetwork/rankine_nn_175_free_startup_2.json") as f:
    data = json.load(f)

## THE CONSTANTS FOR THIS RUN
x = data["market_inputs"]
pmax = x[0]
pmin = x[1]
ramp_rate = x[2]
min_up_time = int(math.ceil(x[3]))
min_down_time = int(math.ceil(x[4]))
marginal_cost = x[5]
fixed_run_cost = x[6]
st_time_hot = x[7]
st_time_warm = x[8]
st_time_cold = x[9]
st_cst_hot = x[10]
st_cst_warm = x[11]
st_cst_cold = x[12]

startup_cost_profile = [(st_time_hot,st_cst_hot),(st_time_warm,st_cst_warm),(st_time_cold,st_cst_cold)]

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

print("PARSED PLUGIN")
