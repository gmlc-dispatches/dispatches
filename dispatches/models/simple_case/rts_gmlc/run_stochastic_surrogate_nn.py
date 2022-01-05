import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
from idaes.core.util import get_solver
import pyomo.environ as pyo
from copy import copy

from model_stochastic_neuralnet_surrogate import stochastic_surrogate_nn_optimization_problem
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
for start_cst_index in range(5):

    #build surrogate design problem
    m =  stochastic_surrogate_nn_optimization_problem(
        heat_recovery=heat_recovery,
        calc_boiler_eff=calc_boiler_eff,
        capital_payment_years=capital_payment_years,
        p_lower_bound=p_max_lower_bound,
        p_upper_bound=p_max_upper_bound,
        plant_lifetime=20,
        include_zone_off = True)


    m.startup_cst.fix(startup_csts[start_cst_index])
    m.pmin_multi.setlb(pmin_lower)

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
    op_cost = []
    op_expr = 0 # in dollars [$]
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
            "start_cost_per_year":startup_expr/1e6,
            "pmax":optimal_p_max
            }
    #write solution
    with open('rankine_results/scikit_surrogate/start_cst_{}_pmin_lower_{}.json'.format(start_cst_index,str(pmin_lower).replace('.','_')), 'w') as outfile:
        json.dump(data, outfile)
