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
from prescient.simulator import Prescient
from types import ModuleType
from argparse import ArgumentParser
from wind_battery_double_loop import MultiPeriodWindBattery
from pathlib import Path
import pyomo.environ as pyo
from idaes.apps.grid_integration import (
    Tracker,
    DoubleLoopCoordinator
)
from idaes.apps.grid_integration.model_data import ThermalGeneratorModelData
from load_parameters import *
from dispatches.workflow.parametrized_bidder import PerfectForecaster
from battery_parametrized_bidder import FixedParametrizedBidder
from dispatches_sample_data import rts_gmlc
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches.case_studies.renewables_case.prescient_options import *

prescient_options = default_prescient_options.copy()

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
    "--battery_energy_capacity",
    dest="battery_energy_capacity",
    help="Set the battery energy capacity in MWh.",
    action="store",
    type=float,
    default=fixed_wind_mw,
)

parser.add_argument(
    "--battery_pmax",
    dest="battery_pmax",
    help="Set the battery power capacity in MW.",
    action="store",
    type=float,
    default=fixed_wind_mw * 0.25,
)

parser.add_argument(
    "--storage_bid",
    dest="storage_bid",
    help="Set the storage bid price in $/MW.",
    action="store",
    type=float,
    default=15.0,
)

options = parser.parse_args()

sim_id = options.sim_id
wind_pmax = options.wind_pmax
battery_energy_capacity = options.battery_energy_capacity
battery_pmax = options.battery_pmax
storage_bid = options.storage_bid
p_min = 0

wind_df = read_rts_gmlc_wind_inputs(rts_gmlc.source_data_path, wind_generator)
wind_df = wind_df[wind_df.index >= start_date]
wind_cfs = wind_df[f"{wind_generator}-RTCF"].values.tolist()

# NOTE: `rts_gmlc_data_dir` should point to a directory containing RTS-GMLC scenarios
rts_gmlc_data_dir = rts_gmlc.source_data_path
output_dir = Path(f"double_loop_parametrized_battery_results")

solver = pyo.SolverFactory(solver_name)

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
    "initial_status": 1,
    "initial_p_output": 0,
    "production_cost_bid_pairs": [(p_min, 0), (wind_pmax, 0)],
    "include_default_p_cost": False,
    "startup_cost_pairs": [(0, 0)],
    "fixed_commitment": None,
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

# PerfectForecaster uses Dataframe with RT and DA wind resource and LMPs. However, for the ParametrizedBidder that
# makes the bid curve based on the `pem_bid` parameter, the LMPs are not needed and are never accessed.
# For non-RE plants, the CFs are never accessed.
forecaster = PerfectForecaster(wind_df)

mp_wind_battery_bid = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=wind_cfs,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
)

bidder_object = FixedParametrizedBidder(
    bidding_model_object=mp_wind_battery_bid,
    day_ahead_horizon=day_ahead_horizon,
    real_time_horizon=real_time_horizon,
    solver=solver,
    forecaster=forecaster,
    storage_marginal_cost=storage_bid,
    storage_mw=battery_pmax
)

################################################################################
################################# Tracker ######################################
################################################################################

mp_wind_battery_track = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=wind_cfs,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
)

# create a `Tracker` using`mp_wind_battery`
tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

mp_wind_battery_track_project = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=wind_cfs,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
)

# create a `Tracker` using`mp_wind_battery`
project_tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track_project,
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

prescient_options["output_directory"] = output_dir
prescient_options["plugin"] = {
    "doubleloop": {
        "module": coordinator.prescient_plugin_module,
        "bidding_generator": wind_generator,
    }
}

Prescient().simulate(**prescient_options)

# write options into the result folder
with open(output_dir / "sim_options_batt_param.json", "w") as f:
    f.write(str(thermal_generator_params))