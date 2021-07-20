import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
from idaes.core.util import get_solver

from stochastic_surrogate import stochastic_surrogate_optimization_problem
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)
import numpy as np
import pandas as pd
from time import perf_counter

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
p_lower_bound = 200
p_upper_bound = 300

build_tic = perf_counter()
m =  stochastic_surrogate_optimization_problem(
    heat_recovery=heat_recovery,
    capital_payment_years=capital_payment_years,
    p_lower_bound=p_lower_bound,
    p_upper_bound=p_upper_bound,
    plant_lifetime=20)
build_toc = perf_counter()

solver = get_solver()
solver.options = {
    "tol": 1e-6
    #"mu_strategy": "adaptive"
}
solver.solve(m, tee=True)

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

model_build_time = build_toc - build_tic
optimal_objective = -value(m.obj)
optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6

w_zone = []
op_cost = []
for i in range(1,11):
    zone = getattr(m, 'zone_{}'.format(i))
    w_zone.append(value(zone.zone_hours))
    op_cost.append(value(zone.fs.operating_cost))

w_zone_weighted = np.array(w_zone)
w_zone_weighted = w_zone_weighted / sum(w_zone_weighted)
op_expr = 0
for i in range(len(w_zone_weighted)):
    op_expr += w_zone_weighted[i]*op_cost[i]

cap_expr = value(m.cap_fs.fs.capital_cost)/capital_payment_years
total_cost = plant_lifetime*op_expr*24*365/1e6 + capital_payment_years*cap_expr
total_revenue = plant_lifetime*value(m.rev_expr)

print("Capital cost:", value(m.cap_fs.fs.capital_cost))
print("Opex cost:", plant_lifetime*op_expr*24*365/1e6 )
print("Revenue: ",total_revenue)

print("The net revenue is M$",total_revenue - total_cost)
print("P_max = ", optimal_p_max, ' MW')
print("Time required to build model= ", model_build_time, "secs")


fig, ax2 = plt.subplots(figsize = (16,8))
ax2.set_xlabel("Power Scenario", fontsize=24)
ax2.set_xticks(range(len(w_zone)))
ax2.tick_params(axis='x', labelrotation = 45)
# ax2.set_xticklabels(["off","90-100%"])
ax2.set_xticklabels(["off","0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"])

ax2.bar(range(len(w_zone)),w_zone_weighted, color="blue")
ax2.set_ylabel("Weight", fontsize=24)
plt.tight_layout()
# ax2.ticklabel_format(useOffset=False, style="plain")

plt.show()
