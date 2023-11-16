# custom plugin file for running the rankine cycle with a battery as a
# multi-period model in the Prescient double-loop
import pickle
import pandas as pd
import pyomo.environ as pyo
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import this_file_dir

from idaes.apps.grid_integration import Tracker
from idaes.apps.grid_integration import Bidder
from idaes.apps.grid_integration import PlaceHolderForecaster
from idaes.apps.grid_integration import DoubleLoopCoordinator

from multiperiod_double_loop_usc import MultiPeriodUsc
from idaes.apps.grid_integration.model_data import GeneratorModelData
import os.path

# with open("usc_gen_data.pkl", "rb") as f:
#     gen_data = pickle.load(f)

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

# default_bid_curve = gen_data["Original Marginal Cost Curve"]
# pmin = gen_data["PMin MW"]
# pmax = gen_data["PMax MW"]
# gen_name = gen_data["generator_name"]
# raise Exception()
tracking_horizon = 12  # hours
bidding_horizon = 48  # hours
n_scenario = 2  # for bidding
n_tracking_hour = 1  # advance n_tracking_hour (i.e. assume we solve every hour)

# create forecaster
price_forecasts_df = pd.read_csv(os.path.join(this_file_dir(), "C:\\grid\\source_code\\idaes-pse\\idaes\\apps\\grid_integration\\examples\\lmp_forecasts_concat.csv"))
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
    # pmin=pmin,
    # pmax=pmax,
    # default_bid_curve=default_bid_curve,
    # generator_name=gen_name,
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
    # pmin=pmin,
    # pmax=pmax,
    # default_bid_curve=default_bid_curve,
    # generator_name=gen_name,
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
    # pmin=pmin,
    # pmax=pmax,
    # default_bid_curve=default_bid_curve,
    # generator_name=gen_name,
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

## Prescient requires the following functions in this module
get_configuration = coordinator.get_configuration
register_plugins = coordinator.register_plugins
