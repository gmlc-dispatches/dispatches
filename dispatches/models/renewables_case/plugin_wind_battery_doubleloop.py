from wind_battery_double_loop import (
    MultiPeriodWindBattery,
    gen_capacity_factor,
    model_data,
    historical_da_prices,
    historical_rt_prices,
)
from idaes.apps.grid_integration import Tracker, SelfScheduler, DoubleLoopCoordinator
from idaes.apps.grid_integration.forecaster import Backcaster
import pyomo.environ as pyo

solver = pyo.SolverFactory("xpress_direct")

################################################################################
################################# bidder #######################################
################################################################################
day_ahead_horizon = 48
real_time_horizon = 4
n_scenario = 2

mp_wind_battery_bid = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=gen_capacity_factor,
)

backcaster = Backcaster(historical_da_prices, historical_rt_prices)

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

tracking_horizon = 48
n_tracking_hour = 1

mp_wind_battery_track = MultiPeriodWindBattery(
    model_data=model_data,
    wind_capacity_factors=gen_capacity_factor,
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
    wind_capacity_factors=gen_capacity_factor,
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

## Prescient requires the following functions in this module
get_configuration = coordinator.get_configuration
register_plugins = coordinator.register_plugins
