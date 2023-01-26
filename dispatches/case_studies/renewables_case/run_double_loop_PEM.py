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
from types import ModuleType
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from dispatches.workflow.coordinator import DoubleLoopCoordinator
from dispatches_sample_data import rts_gmlc
from dispatches.case_studies.renewables_case.wind_PEM_double_loop import MultiPeriodWindPEM
from dispatches.case_studies.renewables_case.load_parameters import *
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches.case_studies.renewables_case.PEM_parametrized_bidder import PEMParametrizedBidder, PerfectForecaster

###
# Script to run a double loop simulation of a Wind + PEM flowsheet (MultiPeriodWindPEM) in Prescient.
# Plant's system dynamics and costs are in wind_PEM_double_loop.
# Plant's bid parameter is "pem_bid" and is constant throughout the simulation.
# Bid curve is a constant function of the available wind resource (one per DA or RT).
# 
# The DoubleLoopCoordinator is from dispatces.workflow.coordinator, not from idaes.apps.grid_integration.
# This version contains some modifications for generator typing and will be merged into idaes in the future.
###

this_file_path = Path(this_file_dir())

usage = "Run double loop simulation with RE model."
parser = ArgumentParser(usage)

parser.add_argument(
    "--sim_id",
    dest="sim_id",
    help="Indicate the simulation ID.",
    action="store",
    type=int,
    default=0,
)

parser.add_argument(
    "--wind_pmax",
    dest="wind_pmax",
    help="Set wind capacity in MW.",
    action="store",
    type=float,
    default=fixed_wind_mw,
)

parser.add_argument(
    "--pem_pmax",
    dest="pem_pmax",
    help="Set the PEM power capacity in MW.",
    action="store",
    type=float,
    default=247.7,
)

parser.add_argument(
    "--pem_bid",
    dest="pem_bid",
    help="Set the PEM bid price in $/MW.",
    action="store",
    type=float,
    default=29.71,
)

parser.add_argument(
    "--reserve_factor",
    dest="reserve_factor",
    help="Set the reserve factor.",
    action="store",
    type=float,
    default=0.15,
)

parser.add_argument(
    "--shortfall",
    dest="shortfall",
    help="Set the shortfall price.",
    action="store",
    type=float,
    default=1000,
)

options = parser.parse_args()

sim_id = options.sim_id
wind_pmax = options.wind_pmax
pem_pmax = options.pem_pmax
pem_bid = options.pem_bid
reserve_factor = options.reserve_factor
shortfall = options.shortfall

p_min = 0
default_wind_bus = 303
bus_name = "Caesar"
wind_generator = "303_WIND_1"
start_date = "01-01-2020"
n_days = 366

# NOTE: `rts_gmlc_data_dir` should point to a directory containing RTS-GMLC scenarios
rts_gmlc_data_dir = rts_gmlc.source_data_path
wind_df = read_rts_gmlc_wind_inputs(rts_gmlc_data_dir, wind_generator)
wind_df = wind_df[wind_df.index >= start_date]
wind_rt_cfs = wind_df[f"{wind_generator}-RTCF"].values.tolist()

output_dir = Path(f"double_loop_parametrized_results_opt")

solver = pyo.SolverFactory("xpress_direct")

thermal_generator_params = {
    "gen_name": wind_generator,
    "bus": bus_name,
    "p_min": p_min,
    "p_max": wind_pmax,
    "min_down_time": 0,
    "min_up_time": 0,
    "ramp_up_60min": wind_pmax,
    "ramp_down_60min": wind_pmax,
    "shutdown_capacity": wind_pmax,
    "startup_capacity": wind_pmax,
    "initial_status": 1,                                        # Has been off for 1 hour before start of simulation
    "initial_p_output": 0,
    "production_cost_bid_pairs": [(p_min, 0), (wind_pmax, 0)],
    "include_default_p_cost": False,
    "startup_cost_pairs": [(0, 0)],
    "fixed_commitment": 1,                                      # Same as the plant in the parameter sweep, which was RE-type and always on
    "spinning_capacity": 0,                                     # Disable participation in some reserve services
    "non_spinning_capacity": 0,
    "supplemental_spinning_capacity": 0,
    "supplemental_non_spinning_capacity": 0,
    "agc_capable": False
}
model_data = ThermalGeneratorModelData(**thermal_generator_params)


################################################################################
################################# bidder #######################################
################################################################################

# same horizons as parameter sweep
day_ahead_horizon = 36                                          
real_time_horizon = 1

# PerfectForecaster uses Dataframe with RT and DA wind resource and LMPs. However, for the ParametrizedBidder that
# makes the bid curve based on the `pem_bid` parameter, the LMPs are not needed and are never accessed.
# For non-RE plants, the CFs are never accessed.
forecaster = PerfectForecaster(wind_df)

mp_wind_pem_bid = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

# ParametrizedBidder is an alternative to the stochastic LMP problem, and serves up a bid curve per timestep
# that is a function of only the bid parameter, PEM capacity and wind resource (DA or RT)
bidder_object = PEMParametrizedBidder(
    bidding_model_object=mp_wind_pem_bid,
    day_ahead_horizon=day_ahead_horizon,
    real_time_horizon=real_time_horizon,
    solver=solver,
    forecaster=forecaster,
    pem_marginal_cost=pem_bid,
    pem_mw=pem_pmax
)

################################################################################
################################# Tracker ######################################
################################################################################

# same tracking_horizon as parameter sweep and n_tracking_hour does not need to be >1
tracking_horizon = 4
n_tracking_hour = 1

mp_wind_pem_track = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

# create a `Tracker` using`mp_wind_battery`
tracker_object = Tracker(
    tracking_model_object=mp_wind_pem_track,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

mp_wind_pem_track_project = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

# create a `Tracker` using`mp_wind_battery`
project_tracker_object = Tracker(
    tracking_model_object=mp_wind_pem_track_project,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

################################################################################
################################# Coordinator ##################################
################################################################################

coordinator = DoubleLoopCoordinator(
    bidder=bidder_object,
    tracker=tracker_object,
    projection_tracker=project_tracker_object,
)

# same options as for prescient sweep
prescient_options = {
    "data_path": rts_gmlc_data_dir,
    "reserve_factor": reserve_factor,
    "simulate_out_of_sample": True,
    "output_directory": output_dir,
    "monitor_all_contingencies":False,
    "input_format": "rts-gmlc",
    "start_date": start_date,
    "num_days": n_days,
    "sced_horizon": real_time_horizon,
    "ruc_mipgap": 0.01,
    "deterministic_ruc_solver": "xpress_persistent",
    "deterministic_ruc_solver_options" : {"threads":2, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
    "sced_solver": "xpress_persistent",
    "sced_frequency_minutes":60,
	    "sced_solver_options" : {"threads":1},
    "ruc_horizon": day_ahead_horizon,
    "compute_market_settlements": True,
    "price_threshold": shortfall,
    "transmission_price_threshold": shortfall / 2,
    "contingency_price_threshold":None,
    "reserve_price_threshold": shortfall / 10,
    "day_ahead_pricing": "aCHP",
    "enforce_sced_shutdown_ramprate":False,
    "ruc_slack_type":"ref-bus-and-branches",    # slack var power balance at reference bus and transmission line flows vs slack var for power balance at every bus
    "sced_slack_type":"ref-bus-and-branches",
    "disable_stackgraphs":True,
    "symbolic_solver_labels": True,
    "plugin": {
        "doubleloop": {
            "module": coordinator.prescient_plugin_module,
            "bidding_generator": wind_generator,
        }
    },
    # verbosity, turn on for debugging
    "output_ruc_solutions": False,
    "write_deterministic_ruc_instances": False,
    "write_sced_instances": False,
    "print_sced": False
}

Prescient().simulate(**prescient_options)

# write options into the result folder
with open(output_dir / "sim_options.json", "w") as f:
    f.write(str(thermal_generator_params.update(prescient_options)))