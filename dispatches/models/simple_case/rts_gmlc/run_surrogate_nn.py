import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
from idaes.core.util import get_solver
import pyomo.environ as pyo
from copy import copy

from model_neuralnet_surrogate import conceptual_design_problem_nn
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


pmin_lower = 0.15
startup_csts = [0., 49.66991167, 61.09068702, 101.4374234,  135.2230393]

#for start_cst_index in range(5):
start_cst_index=2
#build surrogate design problem
m = conceptual_design_problem_nn(
    heat_recovery=heat_recovery,
    calc_boiler_eff=calc_boiler_eff,
    capital_payment_years=capital_payment_years,
    p_lower_bound=p_max_lower_bound,
    p_upper_bound=p_max_upper_bound,
    plant_lifetime=20)


m.startup_cst.fix(startup_csts[start_cst_index])
m.no_load_cst.fix(1.0)
m.min_up_time.fix(4.0)
m.min_dn_multi.fix(1.0)
m.pmin_multi.setlb(pmin_lower)

#turn off marginal cost constraint
m.connect_mrg_cost.deactivate()

solver = get_solver()
solver.options = {
    "tol": 1e-6
    #"mu_strategy": "adaptive"
}
solver.solve(m, tee=True)

print("Revenue Value: ",value(m.revenue))

#get value of surrogate inputs
x = [value(m.pmax),
    value(m.pmin_multi),
    value(m.ramp_multi),
    value(m.min_up_time),
    value(m.min_dn_multi),
    value(m.marg_cst),
    value(m.no_load_cst),
    value(m.startup_cst)
    ]

#calculate revenues, costs, etc...
optimal_objective = -value(m.obj)
optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6

zone_hours = [value(m.zone_off.zone_hours)]
scaled_zone_hours = [value(m.zone_off.scaled_zone_hours)]
op_cost = [value(m.zone_off.fs.operating_cost)]
op_expr = value(m.zone_off.fs.operating_cost) # in dollars [$]
for zone in m.op_zones:
    zone_hours.append(value(zone.zone_hours))
    scaled_zone_hours.append(value(zone.scaled_zone_hours))
    op_cost.append(value(zone.fs.operating_cost))
    op_expr += value(zone.scaled_zone_hours)*value(zone.fs.operating_cost)

revenue_per_year = value(m.revenue)

cap_expr = value(m.cap_fs.fs.capital_cost)/capital_payment_years
startup_expr = value(m.startup_expr)
total_cost = plant_lifetime*op_expr/1e6 + capital_payment_years*cap_expr
total_revenue = plant_lifetime*revenue_per_year

#save surrogate design solution. this can be used for verification.
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
        "nstartups_per_year":value(m.nstartups),
        "start_cost_per_year":startup_expr,
        "pmax":optimal_p_max
        }

#write solution
with open('rankine_results/scikit_surrogate/scikit_verification_1.json'.format(start_cst_index), 'w') as outfile:
    json.dump(data, outfile)
