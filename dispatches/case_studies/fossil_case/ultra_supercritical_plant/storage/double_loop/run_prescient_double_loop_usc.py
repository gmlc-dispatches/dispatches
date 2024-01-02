#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################

from prescient.simulator import Prescient

#point to rts_gmlc scenario data
print("                      ")
print("------Start Prescient Simulation-------")
print("                      ")
# raise Exception()
# rts_gmlc_data_dir = "C:\\grid\\source_code\\Prescient\\downloads\\rts_gmlc\\deterministic_with_network_scenarios"
rts_gmlc_data_dir = "C:\\grid\\source_code\\Prescient\\downloads\\rts_gmlc\\RTS-GMLC\\RTS_Data\\SourceData"

options = {
    "data_path": rts_gmlc_data_dir,
    "input_format": "rts-gmlc",
    "simulate_out_of_sample": True,
    "run_sced_with_persistent_forecast_errors": True,
    "output_directory": "bidding_plugin_test_multiperiod_rankine",
    "start_date": "2020-07-10",
    "num_days": 1,
    "sced_horizon": 12,
    "ruc_horizon": 48,
    "compute_market_settlements": True,
    "day_ahead_pricing": "LMP",
    "ruc_mipgap": 0.01,
    "symbolic_solver_labels": True,
    "reserve_factor": 0.0,
    "deterministic_ruc_solver": "gurobi",
    "output_ruc_solutions": True,
    "sced_solver": "gurobi",
    "print_sced": True,
    # "output_sced_loads": True,
    "enforce_sced_shutdown_ramprate": True,
    "plugin": {
        "doubleloop": {
            "module": "plugin_double_loop_usc.py",
            "bidding_generator": "102_STEAM_3",
        }
    },
}

Prescient().simulate(**options)

# # plot every bid curve over 96 hours
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# costs = bid_profiles_sorted.iloc[:,0:6].to_numpy()
# powers = bid_profiles_sorted.iloc[:,6:-1].to_numpy()
# costs = np.array([costs[i][~np.isnan(costs[i])] for i in range(len(costs))])
# powers = np.array([powers[i][~np.isnan(powers[i])] for i in range(len(powers))])
# times = list(range(len(costs)))

# plt.figure(figsize = (8,8))
# t = np.arange(len(times))
# ax = plt.subplot(projection='3d')
# for t in times:
#     costs_plt = costs[t]
#     powers_plt = powers[t]
#     ax.plot(np.ones(len(costs_plt))*t, powers_plt,costs_plt,color="black")
# ax.set_xlabel("Hour")
# ax.set_ylabel("\nPower [MW]")
# ax.set_zlabel("\n Cost [$/MWh]",linespacing=3.0)
# plt.tight_layout()



# rts_gmlc_data_dir = "/home/xgao1/DowlingLab/RTS-GMLC/RTS_Data/SourceData"
# options = {
#     ​​​​​​"data_path": rts_gmlc_data_dir,
#     "input_format":"rts-gmlc",
#     "simulate_out_of_sample": True,
#     "run_sced_with_persistent_forecast_errors": True,
#     "output_directory": "bidding_plugin_test_multiperiod_rankine",
#     "start_date": "01-02-2020",
#     "num_days": 1,
#     "sced_horizon": 4,
#     "ruc_horizon": 48,
#     "compute_market_settlements": True,
#     "day_ahead_pricing": "LMP",
#     "ruc_mipgap": 0.05,
#     "symbolic_solver_labels": True,
#     "reserve_factor": 0.0,
#     "deterministic_ruc_solver": "gurobi",
#     "sced_solver": "gurobi",
#     "plugin": {​​​​​​
#                "doubleloop": {​​​​​​
#                               "module": "plugin_wind_battery_doubleloop.py",
#                               "bidding_generator": "309_WIND_1",
#                               }​​​​​​
#                }​​​​​​,
#     }​​​​​​

