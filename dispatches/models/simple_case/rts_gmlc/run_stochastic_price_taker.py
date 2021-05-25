import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
# from pyomo.util.infeasible import log_close_to_bounds

# from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver

from simple_rankine_cycle import stochastic_optimization_problem
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=18)
plt.rc('axes', titlesize=18)
import numpy as np
import pandas as pd
from time import perf_counter

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
p_lower_bound = 200
p_upper_bound = 450

# #Load up RTS-GMLC Data for one nominal year.  Use this for prices and dispatch signal
with open('rts_results.npy', 'rb') as f:
    dispatch = np.load(f)
    price = np.load(f)

num_zero = len(price[price < 1])
price_non_zero = price[price > 1]

(n, bins, patches) = plt.hist(price_non_zero, bins=100)
plt.xlabel("LMP $/MWh")
plt.ylabel("Count")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8, 8)


n_with_zero = np.insert(n,0,num_zero)

#lmp_weights = n / sum(n)
lmp_weights = n_with_zero / sum(n_with_zero)
lmp_scenarios = np.array([np.mean([bins[i],bins[i+1]]) for i in range(0,len(bins)-1)])
lmp_scenarios = np.insert(lmp_scenarios,0,0.0)
power_demand = None

build_tic = perf_counter()
m = stochastic_optimization_problem(
    heat_recovery=heat_recovery,
    capital_payment_years=capital_payment_years,
    p_lower_bound=p_lower_bound,
    p_upper_bound=p_upper_bound,
    plant_lifetime=20,
    power_demand=power_demand,
    lmp=lmp_scenarios.tolist(),
    lmp_weights=lmp_weights.tolist())
build_toc = perf_counter()

solver = get_solver()
solver.options = {
    "tol": 1e-6
}
solver.solve(m, tee=True)

op_expr = 0
rev_expr = 0
for i in range(len(lmp_scenarios)):
    scenario = getattr(m, 'scenario_{}'.format(i))
    op_expr += lmp_weights[i]*value(scenario.fs.operating_cost)
    rev_expr += lmp_weights[i]*lmp_scenarios[i]*value(scenario.fs.net_cycle_power_output)/1e6


cap_expr = value(m.cap_fs.fs.capital_cost)/capital_payment_years
total_cost = plant_lifetime*op_expr*24*365/1e6 + capital_payment_years*cap_expr
total_revenue = plant_lifetime*rev_expr*24*365/1e6


print("Capital cost:", value(m.cap_fs.fs.capital_cost))
print("Opex cost:", plant_lifetime*op_expr*24*365/1e6 )
print("Revenue: ",plant_lifetime*rev_expr*24*365/1e6)

# Process results
model_build_time = build_toc - build_tic
optimal_objective = -value(m.obj)
optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6
print("The net revenue is M$", total_revenue - total_cost)
print("P_max = ", optimal_p_max, ' MW')
print("Time required to build model= ", model_build_time, "secs")
p_scenario = []
for i in range(len(lmp_scenarios)):
    scenario = getattr(m, 'scenario_{}'.format(i))
    p_scenario.append(value(scenario.fs.net_cycle_power_output)*1e-6)
p_min = min(p_scenario)


#Try plotting scenario policies?
fig, ax2 = plt.subplots()
# ax.scatter(range(1,len(lmp_scenarios) + 1),lmp_scenarios, color="green")
# # set x-axis label
ax2.set_xlabel("Scenario", fontsize=14)
# # set y-axis label
# ax.set_ylabel("LMP ($/MWh)", color="green", fontsize=14)
#
# ax2 = ax.twinx()
ax2.scatter(range(1,len(lmp_scenarios) + 1),p_scenario, color="blue")
ax2.set_ylabel("Power Produced (MW)", fontsize=14)
ax2.ticklabel_format(useOffset=False, style="plain")
ax2.set_ylim([p_min-5, optimal_p_max+5])

plt.axhline(optimal_p_max, color="red", linestyle="dashed", label="p_max")
plt.axhline(p_min,color="orange", linestyle="dashed", label="p_min")
plt.legend(loc = 'upper left')
plt.grid()
plt.show()
