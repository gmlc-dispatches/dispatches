from pyomo.environ import value
from idaes.core.util import get_solver
import pyomo.environ as pyo

#import alamo conceptual design problem
from model_alamo_surrogate import conceptual_design_problem_alamo
import numpy as np
import pandas as pd
import json


# Inputs for conceptual diesgn problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
calc_boiler_eff = True
p_max_lower_bound = 175
p_max_upper_bound = 450
power_demand = None

m =  conceptual_design_problem_alamo(
    heat_recovery=heat_recovery,
    calc_boiler_eff=calc_boiler_eff,
    capital_payment_years=capital_payment_years,
    p_lower_bound=p_max_lower_bound,
    p_upper_bound=p_max_upper_bound,
    plant_lifetime=20)

#these are representative startup costs based on startup profiles we trained on. 
#you should *not* change these
startup_csts = [0., 49.66991167, 61.09068702, 101.4374234,  135.2230393]
start_cst_index=2
pmin_lower = 0.15

#fix some surrogate inputs
m.startup_cst.fix(startup_csts[start_cst_index])
m.no_load_cst.fix(1.0)
m.min_up_time.fix(4.0)
m.min_dn_multi.fix(1.0)
m.pmin_multi.setlb(pmin_lower)

#sometimes we turn off the marginal cost constraint to emulate the plant bidding whatever it wants.
#normally it would bid the lowest possible marginal cost, but I have seen it exploit overfitting \
#and use an intermediate value.
#m.connect_mrg_cost.deactivate()

solver = get_solver()
solver.options = {
    "tol": 1e-6
    #"mu_strategy": "adaptive"
}
solver.solve(m, tee=True)

print("Revenue Value: ",value(m.revenue))

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
op_expr = value(m.zone_off.fs.operating_cost) 
for zone in m.op_zones:
    zone_hours.append(value(zone.zone_hours))
    scaled_zone_hours.append(value(zone.scaled_zone_hours))
    op_cost.append(value(zone.fs.operating_cost))
    op_expr += value(zone.scaled_zone_hours)*value(zone.fs.operating_cost)
revenue_per_year = value(m.revenue)

#more calculations of revenue and cost
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
with open('rankine_results/alamo_surrogate/conceptual_design_solution_alamo.json', 'w') as outfile:
    json.dump(data, outfile)
