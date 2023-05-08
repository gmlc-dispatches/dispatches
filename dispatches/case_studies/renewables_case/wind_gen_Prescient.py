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
from dispatches_sample_data import rts_gmlc

current_path = os.getcwd()
rtsgmlc_path = rts_gmlc.source_data_path
shortfall = 200
day_ahead_horizon = 36
real_time_horizon = 4
output_path = os.path.join(current_path,f"new_Benchmark_single_wind_gen_sim_15_{shortfall}_rth_{real_time_horizon}")

# default some options
prescient_options = {
        "data_path":rtsgmlc_path,
        "reserve_factor":0.15,
        "simulate_out_of_sample":True,
        "output_directory":output_path,
        "monitor_all_contingencies":False,
        "input_format":"rts-gmlc",
        "start_date":"01-01-2020",
        "num_days":366,
        "sced_horizon":real_time_horizon,
        "ruc_mipgap":0.01,
	"deterministic_ruc_solver": "gurobi",
	"deterministic_ruc_solver_options" : {"threads":4, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
        "sced_solver":"gurobi",
        "sced_frequency_minutes":60,
	    "sced_solver_options" : {"threads":1},
        "ruc_horizon":day_ahead_horizon,
        "compute_market_settlements":True,
        "price_threshold":shortfall,
        "transmission_price_threshold": shortfall / 2,
        "reserve_price_threshold": shortfall / 10,
        "contingency_price_threshold":None,
        "day_ahead_pricing":"aCHP",
        "enforce_sced_shutdown_ramprate":False,
        "ruc_slack_type":"ref-bus-and-branches",
        "sced_slack_type":"ref-bus-and-branches",
	"disable_stackgraphs":True,
        "symbolic_solver_labels":True,
        "output_ruc_solutions": False,
        "write_deterministic_ruc_instances": False,
        "write_sced_instances": False,
        "print_sced":False
        
        }

Prescient().simulate(**prescient_options)
