##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Imports functions from simple_rankine_cycle.py to build the stochastic
problem. This file demonstrates the price take approach.

LMP data set used: ARPA-E FLECCS (NREL)

"""

__author__ = "Jaffer Ghouse"


# Import Pyomo libraries
from pyomo.environ import value
# from pyomo.util.infeasible import log_close_to_bounds

# from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver

from simple_rankine_cycle import stochastic_optimization_problem
from matplotlib import pyplot as plt
import numpy as np
from time import perf_counter

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True

# ARPA-E Signal
lmp_signals = np.load("nrel_scenario_12_rep_days.npy")

# Using single day for all year
# price = lmp_signals[5].tolist()
# weight = 365*np.ones(len(price))
# weight = weight.tolist()
# power_demand = None

# Using 12 representative days - equal weights
# NREL Scenario - Mid NG Price, Carbon Tax 100$, CAISO
price = lmp_signals.flatten().tolist()
weight = (365/12)*np.ones(len(price))
weight = weight.tolist()
power_demand = None

if __name__ == "__main__":

    build_tic = perf_counter()
    m = stochastic_optimization_problem(
        heat_recovery=heat_recovery,
        capital_payment_years=capital_payment_years,
        p_upper_bound=300,
        plant_lifetime=20,
        power_demand=power_demand, lmp=price, lmp_weights=weight)
    build_toc = perf_counter()

    solver = get_solver()
    solver.options = {
        "tol": 1e-6
    }
    solver.solve(m, tee=True)

    # Process results
    model_build_time = build_toc - build_tic
    optimal_objective = -value(m.obj)
    optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6
    print("The net revenue is M$", optimal_objective)
    print("P_max = ", optimal_p_max, ' MW')
    print("Time required to build model= ", model_build_time, "secs")
    p_scenario = []
    for i in range(len(price)):
        scenario = getattr(m, 'scenario_{}'.format(i))
        p_scenario.append(value(scenario    .fs.net_cycle_power_output)*1e-6)
    p_min = min(p_scenario)

    fig, ax = plt.subplots()
    ax.plot(price, color="green")
    # set x-axis label
    ax.set_xlabel("Time (h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("LMP ($/MWh)", color="green", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(p_scenario, color="blue")
    ax2.set_ylabel("Power Produced (MW)", color="blue", fontsize=14)

    plt.axhline(optimal_p_max, color="red", linestyle="dashed", label="p_max")
    plt.axhline(
        p_min,
        color="orange", linestyle="dashed", label="p_min")
    plt.legend()
    plt.grid()
    plt.show()