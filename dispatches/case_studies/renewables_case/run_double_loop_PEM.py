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
from idaes.apps.grid_integration import (
    Tracker,
    DoubleLoopCoordinator
)
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from dispatches_sample_data import rts_gmlc
from dispatches.case_studies.renewables_case.wind_PEM_double_loop import MultiPeriodWindPEM
from dispatches.case_studies.renewables_case.load_parameters import *
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches.case_studies.renewables_case.PEM_parametrized_bidder import PEMParametrizedBidder, PerfectForecaster


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
    default=25.0,
)

parser.add_argument(
    "--pem_bid",
    dest="pem_bid",
    help="Set the PEM bid price in $/MW.",
    action="store",
    type=float,
    default=15.0,
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

# NOTE: `rts_gmlc_data_dir` should point to a directory containing RTS-GMLC scenarios
rts_gmlc_data_dir = rts_gmlc.source_data_path
wind_df = read_rts_gmlc_wind_inputs(rts_gmlc_data_dir, wind_generator)
wind_rt_cfs = wind_df[f"{wind_generator}-RTCF"].values.tolist()

output_dir = Path(f"double_loop_parametrized_results")

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
    "startup_capacity": 0,
    "initial_status": 1,
    "initial_p_output": 0,
    "production_cost_bid_pairs": [(p_min, 0)],
    "startup_cost_pairs": [(0, 0)],
    "fixed_commitment": None,
}
model_data = ThermalGeneratorModelData(**thermal_generator_params)


################################################################################
################################# bidder #######################################
################################################################################
day_ahead_horizon = 48
real_time_horizon = 4

forecaster = PerfectForecaster(wind_df)

mp_wind_pem_bid = MultiPeriodWindPEM(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    pem_pmax_mw=pem_pmax
)

bidder_object = PEMParametrizedBidder(
    bidding_model_object=mp_wind_pem_bid,
    day_ahead_horizon=day_ahead_horizon,
    real_time_horizon=real_time_horizon,
    n_scenario=1,
    solver=solver,
    forecaster=forecaster,
    pem_marginal_cost=pem_bid,
    pem_mw=pem_pmax
)

################################################################################
################################# Tracker ######################################
################################################################################

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


class PrescientPluginModule(ModuleType):
    def __init__(self, get_configuration, register_plugins):
        self.get_configuration = get_configuration
        self.register_plugins = register_plugins


plugin_module = PrescientPluginModule(
    get_configuration=coordinator.get_configuration,
    register_plugins=coordinator.register_plugins,
)

prescient_options = {
    "data_path": rts_gmlc_data_dir,
    "input_format": "rts-gmlc",
    "simulate_out_of_sample": True,
    "run_sced_with_persistent_forecast_errors": True,
    "output_directory": output_dir,
    "monitor_all_contingencies":False,
    "start_date": start_date,
    "num_days": 364,
    "sced_horizon": 1,
    "ruc_horizon": 36,
    "compute_market_settlements": True,
    "day_ahead_pricing": "aCHP",
    "ruc_mipgap": 0.05,
    "symbolic_solver_labels": True,
    "reserve_factor": reserve_factor,
    "price_threshold": shortfall,
    "transmission_price_threshold": shortfall / 2,
    "reserve_price_threshold": shortfall / 10,
    # "deterministic_ruc_solver": "xpress_direct",
    "deterministic_ruc_solver": "xpress_persistent",
    # "deterministic_ruc_solver_options" : {"threads":2, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
    "sced_solver": "xpress_persistent",
    "enforce_sced_shutdown_ramprate":False,
    "ruc_slack_type":"ref-bus-and-branches",
    "sced_slack_type":"ref-bus-and-branches",
    "disable_stackgraphs":True,
    "plugin": {
        "doubleloop": {
            "module": plugin_module,
            "bidding_generator": wind_generator,
        }
    },
}

Prescient().simulate(**prescient_options)

# write options into the result folder
with open(output_dir / "sim_options.json", "w") as f:
    f.write(str(thermal_generator_params.update(prescient_options)))
