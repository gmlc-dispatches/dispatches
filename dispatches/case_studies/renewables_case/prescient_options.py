#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
import pyomo.environ as pyo
from dispatches_sample_data import rts_gmlc
from load_parameters import duration

default_wind_bus = 303
bus_name = "Caesar"
wind_generator = "303_WIND_1"
start_date = "01-01-2020"
n_days = 365
shortfall = 500                                     # 500 $/MWh
reserve_factor = 0.15                               # 15% reserves
rts_gmlc_data_dir = rts_gmlc.source_data_path
day_ahead_horizon = 36
real_time_horizon = duration
tracking_horizon = duration
n_tracking_hour = 1
sim_name = f"re_wind_battery_ratio_duration_sweep_duration_{duration}"
 
solvers_list = ["gurobi_direct", "xpress_direct", "cbc"]

opt = False
for solver_name in solvers_list:
    if pyo.SolverFactory(solver_name).available(exception_flag=False):
        opt = True
        break
if not opt:
    raise RuntimeWarning("No available solvers")

default_prescient_options = {
    "data_path": rts_gmlc_data_dir,
    "reserve_factor": reserve_factor,
    "simulate_out_of_sample": True,
    "output_directory": None,                       # replace in double loop code
    "monitor_all_contingencies":False,
    "input_format": "rts-gmlc",
    "start_date": start_date,
    "num_days": n_days,
    "sced_horizon": real_time_horizon,
    "run_sced_with_persistent_forecast_errors": True,
    "ruc_mipgap": 0.01,
    "deterministic_ruc_solver": solver_name,
    "deterministic_ruc_solver_options" : {"threads":4, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
    "sced_solver": solver_name,
    "sced_frequency_minutes":60,
	    "sced_solver_options" : {"threads":1},
    "ruc_horizon": day_ahead_horizon,
    "compute_market_settlements": True,
    "price_threshold": shortfall,
    "transmission_price_threshold": shortfall / 2,
    "reserve_price_threshold": shortfall / 10,
    "contingency_price_threshold":None,
    "day_ahead_pricing": "aCHP",
    "enforce_sced_shutdown_ramprate":False,
    "ruc_slack_type":"ref-bus-and-branches",    # slack var power balance at reference bus and transmission line flows vs slack var for power balance at every bus
    "sced_slack_type":"ref-bus-and-branches",
    "disable_stackgraphs":True,
    "symbolic_solver_labels": True,
    "plugin": {
        "doubleloop": {
            "module": None,                     # replace in double loop code
            "bidding_generator": None,          # replace in double loop code
        }
    },
    # verbosity, turn on for debugging
    "output_ruc_solutions": False,
    "write_deterministic_ruc_instances": False,
    "write_sced_instances": False,
    "print_sced": False,
    "output_solver_logs": False
}
