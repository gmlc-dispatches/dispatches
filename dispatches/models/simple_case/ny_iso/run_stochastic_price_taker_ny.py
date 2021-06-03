import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
from idaes.core.util import get_solver
from simple_rankine_cycle import stochastic_optimization_problem
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)
import numpy as np
import pandas as pd
from time import perf_counter

capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
p_lower_bound = 200
p_upper_bound = 300

#New York ISO prices
with open('ny_iso_results.npy', 'rb') as f:
    price = np.load(f)

#Plot LMP histogram
(n, bins, patches) = plt.hist(price, bins=100)
plt.xlabel("LMP $/MWh")
plt.ylabel("Count")
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(8, 8)
mean_lmp = np.mean(price)
plt.axvline(mean_lmp, color="green", linestyle="dashed", label="Average LMP",linewidth = 2)
plt.legend(prop={"size":18})
plt.tight_layout()


(n, bins, patches) = plt.hist(price_non_zero, bins=100);
lmp_weights = np.array(n / sum(n))
lmp_scenarios = np.array([np.mean([bins[i],bins[i+1]]) for i in range(0,len(bins)-1)])
#remove the 0 weight scenarios
lmp_scenarios = lmp_scenarios[n != 0]
lmp_weights = lmp_weights[n != 0]


opex_results = []
capex_results = []
rev_results = []
net_rev_results = []
lmp_add = [0,2,4,6,8,15,20,30]
power_demand = None
for i in range(0,len(lmp_add)):
    lmp = lmp_scenarios + lmp_add[i]

    build_tic = perf_counter()
    m = stochastic_optimization_problem(
        heat_recovery=heat_recovery,
        capital_payment_years=capital_payment_years,
        p_lower_bound=p_lower_bound,
        p_upper_bound=p_upper_bound,
        plant_lifetime=20,
        power_demand=power_demand,
        lmp=lmp.tolist(),
        lmp_weights=lmp_weights.tolist())
    build_toc = perf_counter()

    solver = get_solver()
    solver.options = {
        "tol": 1e-8
    }
    solver.solve(m, tee=True)

    op_expr = 0
    rev_expr = 0
    for i in range(len(lmp_scenarios)):
        scenario = getattr(m, 'scenario_{}'.format(i))
        op_expr += lmp_weights[i]*value(scenario.fs.operating_cost)
        rev_expr += lmp_weights[i]*lmp[i]*value(scenario.fs.net_cycle_power_output)/1e6

    cap_expr = value(m.cap_fs.fs.capital_cost)/capital_payment_years
    total_cost = plant_lifetime*op_expr*24*365/1e6 + capital_payment_years*cap_expr
    total_revenue = plant_lifetime*rev_expr*24*365/1e6

    print("Capital cost:", value(m.cap_fs.fs.capital_cost))
    print("Opex cost:", plant_lifetime*op_expr*24*365/1e6)
    print("Revenue: ", plant_lifetime*rev_expr*24*365/1e6)

    # Process results
    model_build_time = build_toc - build_tic
    optimal_objective = -value(m.obj)
    optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6
    print("The net revenue is M$", total_revenue - total_cost)
    print("P_max = ", optimal_p_max, ' MW')
    print("Time required to build model= ", model_build_time, "secs")

    opex_results.append(plant_lifetime*op_expr*24*365/1e6)
    capex_results.append(value(m.cap_fs.fs.capital_cost))
    rev_results.append(plant_lifetime*rev_expr*24*365/1e6)
    net_rev_results.append(total_revenue - total_cost)


p_scenario = []
opex_scenario = []
op_cost = []
for i in range(len(lmp_scenarios)):
    scenario = getattr(m, 'scenario_{}'.format(i))
    p_scenario.append(value(scenario.fs.net_cycle_power_output)*1e-6)
    op_cost.append(value(scenario.fs.operating_cost))
    opex_scenario.append(value(scenario.fs.operating_cost)/value(scenario.fs.net_cycle_power_output*1e-6))
p_min = 0.3*optimal_p_max

#Plot power production vs LMP
fig, ax2 = plt.subplots()
ax2.set_xlabel("Power Produced (MW)", fontsize=18)
ax2.set_ylabel("$/MWh", fontsize=18)
ax2.scatter(p_scenario,lmp, color="blue")
ax2.plot(p_scenario,opex_scenario, color="green",linewidth = 2)
ax2.ticklabel_format(useOffset=False, style="plain")
plt.legend(["Operating Cost","LMP"],loc = 'upper left')
plt.grid()
plt.show()
