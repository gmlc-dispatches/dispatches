##############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
##############################################################################

"""
This script uses the multiperiod object for the integrated fossil case study
to set up and run the double-loop framework.
"""

__author__ = "Naresh Susarla"

import os
from types import ModuleType
import pandas as pd
from importlib import resources
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
try:
    from importlib import resources  # Python 3.8+
except ImportError:
    import importlib_resources as resources  # Python 3.7

# Import Pyomo packages
import pyomo.environ as pyo

# Import IDAES packages
from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration import Bidder
from idaes.apps.grid_integration import PlaceHolderForecaster
from idaes.apps.grid_integration import DoubleLoopCoordinator
# from idaes.apps.grid_integration.model_data import GeneratorModelData
from idaes.apps.grid_integration.model_data import (
    GeneratorModelData as _UpstreamGeneratorModelData,
    RealValueValidator,
    AtLeastPminValidator
)

# Import Prescient simulator
from prescient.simulator import Prescient

# Import integrated ultra-supercritical power plant with energy storage model
from dispatches_sample_data import rts_gmlc
from dispatches.models.fossil_case.ultra_supercritical_plant import storage
from dispatches.models.fossil_case.ultra_supercritical_plant.storage.multiperiod_double_loop_usc import MultiPeriodUsc

class GeneratorModelData(_UpstreamGeneratorModelData):
    p_min_agc = RealValueValidator(min_val=0)
    p_max_agc = AtLeastPminValidator()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p_min_agc = self.p_min
        self.p_max_agc = self.p_max

generator_data = {
    "gen_name": "102_STEAM_3",
    "generator_type": "thermal",
    "p_min": 286,
    "p_max": 460,
    "min_down_time": 4,
    "min_up_time": 8,
    "ramp_up_60min": 60,
    "ramp_down_60min": 60,
    "shutdown_capacity": 286,
    "startup_capacity": 286,
    "production_cost_bid_pairs": [
        (286, 22.16602),
        (350, 19.0723),
        (400, 18.29703),
        (430, 17.71727),
        (460, 17.71727),
    ],
    "startup_cost_pairs": [(4, 7355.42), (10, 10488.35), (12, 11383.41)],
    "fixed_commitment": 1,
}
model_data = GeneratorModelData(**generator_data)

tracking_horizon = 4  # hours
bidding_horizon = 48  # hours
n_scenario = 2  # for bidding
n_tracking_hour = 1  # advance n_tracking_hour (i.e. assume we solve every hour)
num_days = 2

# create forecaster
# price_forecasts_df = pd.read_csv(
#     os.path.join(
#         this_file_dir(),
#         "C:\\grid\\source_code\\idaes-pse\\idaes\\apps\\grid_integration\\examples\\lmp_forecasts_concat.csv"))

with resources.open_text("idaes.apps.grid_integration.examples", "lmp_forecasts_concat.csv") as f:
    price_forecasts_df = pd.read_csv(f)
forecaster = PlaceHolderForecaster(price_forecasts_df=price_forecasts_df)

# create solver
solver = pyo.SolverFactory("ipopt")
solver.options = {
    "max_iter": 200,
}


# Setup trackers, bidder, and coordinator
#################################################################################
# Tracker
mp_usc_tracker = MultiPeriodUsc(
    horizon=tracking_horizon,
    model_data=model_data
)

thermal_tracker = Tracker(
    tracking_model_object=mp_usc_tracker,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

# Projection Tracker
mp_usc_projection_tracker = MultiPeriodUsc(
    horizon=tracking_horizon,
    model_data=model_data
)

thermal_projection_tracker = Tracker(
    tracking_model_object=mp_usc_projection_tracker,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

# Bidder
mp_usc_bidder = MultiPeriodUsc(
    horizon=bidding_horizon,
    model_data=model_data
)
thermal_bidder = Bidder(
    bidding_model_object=mp_usc_bidder,
    n_scenario=n_scenario,
    solver=solver,
    forecaster=forecaster,
)

# Coordinator
coordinator = DoubleLoopCoordinator(
    bidder=thermal_bidder,
    tracker=thermal_tracker,
    projection_tracker=thermal_projection_tracker,
)


class PrescientPluginModule(ModuleType):
    def __init__(self, get_configuration, register_plugins):
        self.get_configuration = get_configuration
        self.register_plugins = register_plugins


plugin_module = PrescientPluginModule(
    get_configuration=coordinator.get_configuration,
    register_plugins=coordinator.register_plugins,
)


# If installing the dispatches-sample-data
rts_gmlc_data_dir = str(rts_gmlc.source_data_path)
# rts_gmlc_data_dir = "C:\\grid\\source_code\\Prescient\\downloads\\rts_gmlc\\RTS-GMLC\\RTS_Data\\SourceData"

options = {
    "data_path": rts_gmlc_data_dir,
    "input_format": "rts-gmlc",
    "simulate_out_of_sample": True,
    "run_sced_with_persistent_forecast_errors": True,
    "output_directory": "bidding_multiperiod_usc",
    "start_date": "2020-07-10",
    "num_days": num_days,
    "sced_horizon": tracking_horizon,
    "ruc_horizon": bidding_horizon,
    "compute_market_settlements": True,
    "day_ahead_pricing": "LMP",
    "ruc_mipgap": 0.05,
    "symbolic_solver_labels": True,
    "reserve_factor": 0.0,
    "deterministic_ruc_solver": "gurobi",
    "output_ruc_solutions": True,
    "sced_solver": "gurobi",
    "print_sced": True,
    "enforce_sced_shutdown_ramprate": True,
    "plugin": {
        "doubleloop": {
            "module": plugin_module,
            "bidding_generator": "102_STEAM_3",
        }
    },
}

Prescient().simulate(**options)
