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

# custom plugin file for running the rankine cycle with a battery as a
# multi-period model in the Prescient double-loop
import pickle
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import this_file_dir

from idaes.apps.grid_integration.examples.utils import (
    rts_gmlc_generator_dataframe,
    rts_gmlc_bus_dataframe,
    prescient_5bus,
    daily_da_price_means,
    daily_rt_price_means,
    daily_da_price_stds,
    daily_rt_price_stds,
)

from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration import Bidder
from idaes.apps.grid_integration import PlaceHolderForecaster
from idaes.apps.grid_integration import DoubleLoopCoordinator

from multiperiod_double_loop_usc import MultiPeriodUsc
from idaes.apps.grid_integration.model_data import GeneratorModelData
from idaes.apps.grid_integration.model_data import (
    GeneratorModelData as _UpstreamGeneratorModelData,
    RealValueValidator,
    AtLeastPminValidator
)
import os.path

# with open("usc_gen_data.pkl", "rb") as f:
#     gen_data = pickle.load(f)

generator_data = {
    "gen_name": "102_STEAM_3",
    "bus": "Carter",
    "p_min": 286,
    "p_max": 460,
}
model_data = GeneratorModelData(**generator_data)

# default_bid_curve = gen_data["Original Marginal Cost Curve"]
# pmin = gen_data["PMin MW"]
# pmax = gen_data["PMax MW"]
# gen_name = gen_data["generator_name"]
# raise Exception()
tracking_horizon = 12  # hours
bidding_horizon = 48  # hours
day_ahead_horizon = 48  # hours
real_time_horizon = 4  # hours
n_scenario = 2  # for bidding
n_tracking_hour = 1  # advance n_tracking_hour (i.e. assume we solve every hour)

# create forecasterprice_forecasts_df = pd.read_csv(os.path.join(this_file_dir(), "C:\\grid\\source_code\\idaes-pse\\idaes\\apps\\grid_integration\\examples\\lmp_forecasts_concat.csv"))
forecaster = PlaceHolderForecaster(
    daily_da_price_means=daily_da_price_means,
    daily_rt_price_means=daily_rt_price_means,
    daily_da_price_stds=daily_da_price_stds,
    daily_rt_price_stds=daily_rt_price_stds,
)
# create solver
solver = pyo.SolverFactory("ipopt")
solver.options = {
    "max_iter": 200,
}


# Setup trackers, bidder, and coordinator
#################################################################################
# Tracker
mp_usc_tracker = MultiPeriodUsc(
    model_data=model_data
)

thermal_tracker = Tracker(
    tracking_model_object=mp_usc_tracker,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

# Projection Tracker
mp_usc_projection_tracker = MultiPeriodUsc(
    model_data=model_data
)

thermal_projection_tracker = Tracker(
    tracking_model_object=mp_usc_projection_tracker,
    tracking_horizon=tracking_horizon,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

# Bidder
mp_usc_bidder = MultiPeriodUsc(
    model_data=model_data
)
thermal_bidder = Bidder(
    bidding_model_object=mp_usc_bidder,
    day_ahead_horizon=day_ahead_horizon,
    real_time_horizon=real_time_horizon,
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

## Prescient requires the following functions in this module
get_configuration = coordinator.get_configuration
register_plugins = coordinator.register_plugins
