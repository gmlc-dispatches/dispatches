import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
# from pyomo.util.infeasible import log_close_to_bounds

# from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver

from stochastic_surrogate import stochastic_surrogate_optimization_problem
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
p_upper_bound = 300


build_tic = perf_counter()
m =  stochastic_surrogate_optimization_problem(
    heat_recovery=heat_recovery,
    capital_payment_years=capital_payment_years,
    p_upper_bound=p_upper_bound,
    plant_lifetime=20)
build_toc = perf_counter()

solver = get_solver()
solver.options = {
    "tol": 1e-6,
    "mu_strategy": "adaptive"
}
solver.solve(m, tee=True)

model_build_time = build_toc - build_tic
optimal_objective = -value(m.obj)
optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6
print("The net revenue is M$", optimal_objective)
print("P_max = ", optimal_p_max, ' MW')
print("Time required to build model= ", model_build_time, "secs")


# fig, ax = plt.subplots()
# ax.plot(price, color="green")
# # set x-axis label
# ax.set_xlabel("Time (h)", fontsize=14)
# # set y-axis label
# ax.set_ylabel("LMP ($/MWh)", color="green", fontsize=14)
#
# ax2 = ax.twinx()
# ax2.plot(p_scenario, color="blue")
# ax2.set_ylabel("Power Produced (MW)", color="blue", fontsize=14)
# ax2.ticklabel_format(useOffset=False, style="plain")
# ax2.set_ylim([p_min-5, optimal_p_max+5])
#
# plt.axhline(optimal_p_max, color="red", linestyle="dashed", label="p_max")
# plt.axhline(
#     p_min,
#     color="orange", linestyle="dashed", label="p_min")
# plt.legend()
# plt.grid()
# plt.show()
