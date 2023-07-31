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

import os
from prescient.simulator import Prescient


def download_rts_gmlc(dir_name="DISPATCHES_RTS-GMLC"):
    cur_path = os.getcwd()
    this_file_path = os.path.dirname(os.path.realpath(__file__))

    if os.path.isdir(os.path.join(this_file_path, dir_name)):
        return
    
    os.chdir(this_file_path)
    os.system(f"git clone --depth=1 https://github.com/bknueven/RTS-GMLC -b no_reserves {dir_name}")
    
    os.chdir(cur_path)


download_rts_gmlc()
this_file_path = os.path.dirname(os.path.realpath(__file__))

start_date = "01-01-2020"
n_days = 366
shortfall = 500                                     # 500 $/MWh
reserve_factor = 0.15                               # 15% reserves
day_ahead_horizon = 36
real_time_horizon = 1
tracking_horizon = 4
n_tracking_hour = 1
output_dir = "ne_without_pem_results"

# default some options
prescient_options = {
    "data_path": os.path.join(this_file_path, "DISPATCHES_RTS-GMLC/RTS_Data/SourceData"),
    "reserve_factor": reserve_factor,
    "simulate_out_of_sample": True,
    "output_directory": output_dir,
    "monitor_all_contingencies": False,
    "input_format": "rts-gmlc",
    "start_date": start_date,
    "num_days": n_days,
    "sced_horizon": real_time_horizon,
    "ruc_mipgap": 0.01,
    "deterministic_ruc_solver": "gurobi_persistent",
    "deterministic_ruc_solver_options" : {
        "threads": 4, 
        "heurstrategy": 2, 
        "cutstrategy": 3, 
        "symmetry": 2,
        "maxnode": 1000,
    },
    "sced_solver": "gurobi_persistent",
    "sced_frequency_minutes": 60,
    "sced_solver_options" : {"threads": 1},
    "ruc_horizon": day_ahead_horizon,
    "compute_market_settlements": True,
    "output_solver_logs": False,
    "price_threshold": shortfall,
    "transmission_price_threshold": shortfall / 2,
    "contingency_price_threshold": None,
    "reserve_price_threshold": shortfall / 10,
    "day_ahead_pricing": "aCHP",
    "enforce_sced_shutdown_ramprate": False,
    "ruc_slack_type": "ref-bus-and-branches",
    "sced_slack_type": "ref-bus-and-branches",
    "disable_stackgraphs": True,
}


Prescient().simulate(**prescient_options)
