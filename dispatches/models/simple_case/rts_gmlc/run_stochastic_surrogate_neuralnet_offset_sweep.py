import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
from idaes.core.util import get_solver
import pyomo.environ as pyo
from copy import copy

from stochastic_neuralnet_surrogate import stochastic_surrogate_nn_optimization_problem
import numpy as np
import pandas as pd
from time import perf_counter
import json

from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
calc_boiler_eff = True
p_max_lower_bound = 175
p_max_upper_bound = 450
power_demand = None
fix_startup_profile = True



revenue_offsets = [0.0,10.0,20.0,30.0]

m =  stochastic_surrogate_nn_optimization_problem(
    heat_recovery=heat_recovery,
    calc_boiler_eff=calc_boiler_eff,
    capital_payment_years=capital_payment_years,
    p_lower_bound=p_max_lower_bound,
    p_upper_bound=p_max_upper_bound,
    plant_lifetime=20,
    fix_startup_profile = fix_startup_profile,
    include_zone_off = True)

rev_expr = copy(m.revenue.expr)

data_results = []

for offset in revenue_offsets:
    m.revenue.set_value(expr = rev_expr + offset)

    solver = get_solver()
    solver.options = {
        "tol": 1e-6
        #"mu_strategy": "adaptive"
    }
    solver.solve(m, tee=True)

    print("Revenue Value: ",value(m.revenue))

    x = [value(m.pmax),value(m.pmin),value(m.ramp_rate),
        value(m.min_up_time),
        value(m.min_down_time),
        value(m.marg_cst),
        value(m.no_load_cst),
        value(m.st_time_hot),
        value(m.st_time_warm),
        value(m.st_time_cold),
        value(m.st_cst_hot),
        value(m.st_cst_warm),
        value(m.st_cst_cold)]

    optimal_objective = -value(m.obj)
    optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6

    zone_hours = [value(m.zone_off.zone_hours)]
    scaled_zone_hours = [value(m.zone_off.scaled_zone_hours)]
    op_cost = []
    op_expr = 0 # in dollars [$]
    for zone in m.op_zones:
        zone_hours.append(value(zone.zone_hours))
        scaled_zone_hours.append(value(zone.scaled_zone_hours))
        op_cost.append(value(zone.fs.operating_cost))
        op_expr += value(zone.scaled_zone_hours)*value(zone.fs.operating_cost)

    revenue_per_year = value(m.revenue)

    cap_expr = value(m.cap_fs.fs.capital_cost)/capital_payment_years
    total_cost = plant_lifetime*op_expr/1e6 + capital_payment_years*cap_expr
    total_revenue = plant_lifetime*revenue_per_year

    #SAVE SURROGATE SOLUTION
    data = {"market_inputs":x,
            "revenue_surrogate":value(m.rev_surrogate),
            "revenue_rankine":revenue_per_year,
            "scaled_dispatch_zones":scaled_zone_hours,
            "dispatch_zones":zone_hours,
            "operating_cost":op_cost,
            "capital_cost":value(m.cap_fs.fs.capital_cost),
            "total_revenue":total_revenue,
            "total_cost":total_cost,
            "net_revenue":total_revenue - total_cost,
            "opex_per_year":op_expr/1e6,
            "pmax":optimal_p_max
            }
    data_results.append(data)


with open('results_solutions_neuralnetwork/rankine_nn_{}_fix_startup_profile_blue_offsets.json'.format(p_max_lower_bound), 'w') as outfile:
    json.dump(data_results, outfile)

#Plot revenue offset results
#Load up the json file to read parameters
with open("results_solutions_neuralnetwork/rankine_nn_175_fix_startup_profile_blue_offsets.json","r") as f:
    data_results = json.load(f)

net_revenue = [data_results[i]["net_revenue"] for i in range(len(revenue_offsets))]

fig, ax = plt.subplots(figsize = (8,8))
ax.set_xlabel("Additional Yearly Revenue [MM$/yr]")
ax.set_ylabel("20 Year Net Revenue [MM$]")
ax.plot(revenue_offsets,net_revenue,linewidth = 2.0)

plt.tight_layout()
fig.savefig("results_solutions_neuralnetwork/additional_revenue.png")
