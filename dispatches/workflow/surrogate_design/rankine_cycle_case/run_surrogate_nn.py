#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

#this script runs the conceptual design problem using neural network surrogates
#for revenue, time spent in different power outputs, and the number of plant startups

# Import Pyomo libraries
from pyomo.environ import value
from idaes.core.util import get_solver
import pyomo.environ as pyo
from copy import copy

#import the neural network conceptual design problem
from model_neuralnet_surrogate import conceptual_design_problem_nn
import numpy as np
import pandas as pd
from time import perf_counter
import json


# Inputs for design problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
calc_boiler_eff = True  #I think this always needs to be True to work with the surrogate
p_max_lower_bound = 175
p_max_upper_bound = 450
power_demand = None #the design problem can use a dispatch signal in the objective


#build the surrogate design problem
m = conceptual_design_problem_nn(
    heat_recovery=heat_recovery,
    calc_boiler_eff=calc_boiler_eff,
    capital_payment_years=capital_payment_years,
    p_lower_bound=p_max_lower_bound,
    p_upper_bound=p_max_upper_bound,
    plant_lifetime=20,
    coal_price=50)

#setup surrogate inputs
pmin_lower = 0.15 #lower bound on pmin_multiplier

#these are representative startup costs based on startup profiles we trained on.
#you should *not* change these
startup_csts = [0., 49.66991167, 61.09068702, 101.4374234,  135.2230393]

#fix some surrogate inputs
start_cst_index=2
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
status = solver.solve(m, tee=True)
sol_time = status['Solver'][0]['Time']

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

#more calculations of revenue and cost
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
        "pmax":optimal_p_max,
        "solution_time":sol_time
        }

#write solution
with open('design_results/scikit_surrogate/conceptual_design_solution_nn_2.json', 'w') as outfile:
    json.dump(data, outfile)
