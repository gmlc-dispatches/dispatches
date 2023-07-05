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
from argparse import ArgumentParser
from wind_battery_double_loop import MultiPeriodWindBattery
from idaes.apps.grid_integration import (
    Tracker,
    Bidder,
    SelfScheduler,
)
from dispatches.workflow.coordinator import DoubleLoopCoordinator
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.model_data import (
    RenewableGeneratorModelData,
    ThermalGeneratorModelData,
)
from idaes import __version__
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
import pandas as pd
from pathlib import Path
from dispatches_data.api import path
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
    default=200.0,
)

parser.add_argument(
    "--battery_energy_capacity",
    dest="battery_energy_capacity",
    help="Set the battery energy capacity in MWh.",
    action="store",
    type=float,
    default=100.0,
)

parser.add_argument(
    "--battery_pmax",
    dest="battery_pmax",
    help="Set the battery power capacity in MW.",
    action="store",
    type=float,
    default=25.0,
)

parser.add_argument(
    "--n_scenario",
    dest="n_scenario",
    help="Set the number of price scenarios.",
    action="store",
    type=int,
    default=3,
)

parser.add_argument(
    "--reserve_factor",
    dest="reserve_factor",
    help="Set the reserve factor.",
    action="store",
    type=float,
    default=0.0,
)

parser.add_argument(
    "--participation_mode",
    dest="participation_mode",
    help="Indicate the market participation mode.",
    action="store",
    type=str,
    default="Bid",
)

options = parser.parse_args()

sim_id = options.sim_id
wind_pmax = options.wind_pmax
battery_energy_capacity = options.battery_energy_capacity
battery_pmax = options.battery_pmax
n_scenario = options.n_scenario
participation_mode = options.participation_mode
reserve_factor = options.reserve_factor

allowed_participation_modes = {"Bid", "SelfSchedule"}
if participation_mode not in allowed_participation_modes:
    raise ValueError(
        f"The provided participation mode {participation_mode} is not supported."
    )

p_min = 0

wind_df = read_rts_gmlc_wind_inputs(path("rts_gmlc") / "SourceData", wind_generator)
wind_df = wind_df[wind_df.index >= start_date]
wind_rt_cfs = wind_df[f"{wind_generator}-RTCF"].values.tolist()

output_dir = Path(f"sim_{sim_id}_results")

solver = pyo.SolverFactory(solver_name)

if participation_mode == "Bid":
    thermal_generator_params = {
        "gen_name": wind_generator,
        "bus": bus_name,
        "p_min": p_min,
        "p_max": wind_pmax,
        "min_down_time": 0,
        "min_up_time": 0,
        "ramp_up_60min": wind_pmax + battery_pmax,
        "ramp_down_60min": wind_pmax + battery_pmax,
        "shutdown_capacity": wind_pmax + battery_pmax,
        "startup_capacity": wind_pmax + battery_pmax,
        "initial_status": 1,
        "initial_p_output": 0,
        "production_cost_bid_pairs": [(p_min, 0), (wind_pmax + battery_pmax, 0)],
        "include_default_p_cost": False,
        "startup_cost_pairs": [(0, 0)],
        "fixed_commitment": None,
    }
    model_data = ThermalGeneratorModelData(**thermal_generator_params)
elif participation_mode == "SelfSchedule":
    generator_params = {
        "gen_name": wind_generator,
        "bus": bus_name,
        "p_min": p_min,
        "p_max": wind_pmax,
        "p_cost": 0,
        "fixed_commitment": None,
    }
    model_data = RenewableGeneratorModelData(**generator_params)

historical_da_prices = {
    bus_name: [
        19.983547,
        0.0,
        0.0,
        19.983547,
        21.647258,
        21.647258,
        33.946708,
        21.647258,
        0.0,
        0.0,
        19.983547,
        20.846138,
        20.419098,
        21.116411,
        21.116411,
        21.843654,
        33.752662,
        27.274616,
        27.274616,
        26.324557,
        23.128644,
        21.288154,
        21.116714,
        21.116714,
    ]
}
historical_rt_prices = {
    bus_name: [
        30.729141,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        32.451804,
        34.888412,
        0.0,
        0.0,
        0.0,
        0.0,
        19.983547,
        21.116411,
        19.034775,
        16.970947,
        20.419098,
        26.657418,
        25.9087,
        24.617414,
        24.617414,
        22.492854,
        10000.0,
        23.437807,
    ]
}

################################################################################
################################# bidder #######################################
################################################################################

mp_wind_battery_bid = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
    wind_pmax_mw=wind_pmax,
    battery_pmax_mw=battery_pmax,
    battery_energy_capacity_mwh=battery_energy_capacity,
)

backcaster = Backcaster(historical_da_prices, historical_rt_prices)

if participation_mode == "Bid":
    bidder_object = Bidder(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        n_scenario=n_scenario,
        solver=solver,
        forecaster=backcaster,
    )
elif participation_mode == "SelfSchedule":
    bidder_object = SelfScheduler(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        n_scenario=n_scenario,
        solver=solver,
        forecaster=backcaster,
    )

################################################################################
################################# Tracker ######################################
################################################################################

mp_wind_battery_track = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=wind_rt_cfs,
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
    wind_capacity_factors=wind_rt_cfs,
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
with open(output_dir / "sim_options.txt", "w") as f:
    f.write(str(options))
