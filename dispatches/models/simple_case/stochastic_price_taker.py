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

LMP data set used:
1. ARPA-E FLECCS (NREL)
2. RTS-GMLC (DISPATCHES)
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
import pandas as pd
from time import perf_counter

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
calc_boiler_eff = False
p_max_lower_bound = 175
p_max__upper_bound = 450

# ARPA-E Signal - NREL
# NREL Scenario - Mid NG Price, Carbon Tax 100$, CAISO
# average_hourly = np.load("nrel_scenario_average_hourly.npy")
# rep_days = np.load("nrel_scenario_12_rep_days.npy")
# weights_rep_days = np.load("nrel_scenario_12_rep_days_weights.npy")
# raw_data = pd.read_pickle("nrel_raw_data_to_pickle.pkl")

# RTS-GMLC Signal Unfiltered
# average_hourly = np.load("rts_bus_scenario_average_hourly.npy")
# rep_days = np.load("rts_bus_scenario_12_rep_days.npy")
# weights_rep_days = np.load("rts_bus_scenario_12_rep_days_weights.npy")
raw_data = pd.read_pickle("rts_raw_data_to_pickle.pkl")

# RTS-GMLC Signal Filtered < 100
# average_hourly = np.load("rts_bus_scenario_average_hourly.npy")
# rep_days = np.load("rts_bus_scenario_12_rep_days.npy")
# weights_rep_days = np.load("rts_bus_scenario_12_rep_days_weights.npy")
# raw_data = pd.read_pickle("rts_raw_data_filtered_100_to_pickle.pkl")

# Using average_hourly for single day for all year
# price = average_hourly.tolist()
# weight = 365*np.ones(len(price))
# weight = weight.tolist()
# power_demand = None

# Using 12 representative days - equal weights
# price = rep_days.flatten().tolist()
# ones_array = np.ones((len(rep_days), 24))
# for i in range(0, len(rep_days)):
#     ones_array[i] = weights_rep_days[i]*ones_array[i]
# weight = ones_array.flatten().tolist()
# power_demand = None

# Using 365 representative days - equal weights
# price_all = raw_data["MiNg_$100_CAISO"].tolist()
# filtered price; exculde LMPs < 10$/MWh
# price = list(filter(lambda i: i >= 10, price_all))
# price = price_all

# RTS dataset
price = raw_data["Price"].tolist()
ones_array = np.ones(len(price))
weight = ones_array.flatten().tolist()
power_demand = None

if __name__ == "__main__":

    build_tic = perf_counter()
    m = stochastic_optimization_problem(
        heat_recovery=heat_recovery,
        calc_boiler_eff=calc_boiler_eff,
        capital_payment_years=capital_payment_years,
        p_max_lower_bound=p_max_lower_bound,
        p_max_upper_bound=p_max__upper_bound,
        plant_lifetime=20,
        power_demand=power_demand, lmp=price, lmp_weights=weight)
    build_toc = perf_counter()

    solver = get_solver()
    solver.options = {
        "tol": 1e-6
    }
    res = solver.solve(m, tee=True)

    # Process results
    model_build_time = build_toc - build_tic
    optimal_objective = -value(m.obj)
    optimal_p_max = value(m.cap_fs.fs.net_cycle_power_output)*1e-6

    p_scenario = []
    p_max_scenario = []
    op_cost_scenario = []
    cycle_eff_scenario = []
    boiler_eff_scenario = []
    for i in range(len(price)):
        scenario = getattr(m, 'scenario_{}'.format(i))
        p_scenario.append(value(scenario.fs.net_cycle_power_output)*1e-6)
        # p_max_scenario.append(value(scenario.fs.net_power_max)*1e-6)
        cycle_eff_scenario.append(value(scenario.fs.cycle_efficiency))
        boiler_eff_scenario.append(value(scenario.fs.boiler_eff))
        op_cost_scenario.append(value(scenario.fs.operating_cost))
    p_min = 0.3*max(p_scenario)

    # calculate operating cost per MWh
    op_cost = []
    for i in range(len(p_scenario)):
        op_cost.append(op_cost_scenario[i]/p_scenario[i])

    print("The net revenue is M$", optimal_objective/1e6)
    print("P_max = ", optimal_p_max, ' MW')
    print("P_min = ", p_min, ' MW')
    print("Time required to build model= ", model_build_time, "secs")

    # print()
    # print("operating cost $/MWh")
    # print(op_cost)
    # print()
    # print("cycle efficiency")
    # print(cycle_eff_scenario)
    # print()
    # print("boiler efficiency")
    # print(boiler_eff_scenario)

    hour_list = list(range(1, len(price) + 1))
    fig, ax = plt.subplots()
    ax.step(hour_list, price, linestyle="dashed", color="green")
    # ax.step(hour_list, op_cost, color='green', marker='o',
    #         linestyle="None",
    #         markerfacecolor="None",
    #         markersize=5,
    #         label="operating cost")
    # set x-axis label
    ax.set_xlabel("Time (h)", fontsize=14)
    # set y-axis label
    ax.set_ylabel("LMP ($/MWh)", color="green", fontsize=14)

    ax2 = ax.twinx()
    ax2.step(hour_list, p_scenario, color="blue", marker='o',
             linestyle="None",
             markerfacecolor="None",
             markersize=5)
    ax2.set_ylabel("Power Produced (MW)", color="blue",fontsize=14)
    ax2.ticklabel_format(useOffset=False, style="plain")
    ax2.set_ylim([p_min-5, optimal_p_max+25])

    plt.axhline(optimal_p_max, color="red",
                linewidth=0.9, label="p_max")
    plt.axhline(
        p_min,
        color="orange", linewidth=0.9, label="p_min")
    plt.xlim(1, len(price))
    ax.grid(which='major', axis='both', linestyle='--')
    ax.legend()
    ax2.legend()
    plt.savefig("RTS_365_days_without_boiler_eff.png")
    plt.show()

    # fig2, ax = plt.subplots()
    # ax.step(hour_list, op_cost, 'ro', label="operating cost")
    # ax.step(hour_list, price, color="green", label="LMP")
    # ax.set_xlabel("Time (h)", fontsize=14)
    # ax.set_ylabel("$/MWh", fontsize=14)
    # plt.legend()
    # ax.grid(which='major', axis='both', linestyle='--')
    # plt.show()
    # plt.savefig("arpae_rep_days_with_boiler_eff.png")

