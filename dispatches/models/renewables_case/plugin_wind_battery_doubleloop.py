from wind_battery_double_loop import (
    MultiPeriodWindBattery,
    SimpleForecaster,
    gen_capacity_factor,
    default_wind_bus,
    wind_generator,
)
from idaes.apps.grid_integration import Tracker, SelfScheduler, DoubleLoopCoordinator
import pyomo.environ as pyo

solver = pyo.SolverFactory("gurobi")
solver.options["NonConvex"] = 2
pmin = 0
pmax = 200

################################################################################
################################# bidder #######################################
################################################################################
bidding_horizon = 48
n_scenario = 1

mp_wind_battery_bid = MultiPeriodWindBattery(
    horizon=bidding_horizon,
    pmin=pmin,
    pmax=pmax,
    generator_name=wind_generator,
    wind_capacity_factors=gen_capacity_factor,
)

price_forecaster = SimpleForecaster(
    horizon=bidding_horizon, n_sample=n_scenario, bus=default_wind_bus
)

bidder_object = SelfScheduler(
    bidding_model_object=mp_wind_battery_bid,
    n_scenario=n_scenario,
    horizon=bidding_horizon,
    solver=solver,
    forecaster=price_forecaster,
)

################################################################################
################################# Tracker ######################################
################################################################################

tracking_horizon = 4
n_tracking_hour = 1

mp_wind_battery_track = MultiPeriodWindBattery(
    horizon=tracking_horizon,
    pmin=pmin,
    pmax=pmax,
    generator_name=wind_generator,
    wind_capacity_factors=gen_capacity_factor,
)

# create a `Tracker` using`mp_wind_battery`
tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track,
    n_tracking_hour=n_tracking_hour,
    solver=solver,
)

mp_wind_battery_track_project = MultiPeriodWindBattery(
    horizon=tracking_horizon,
    pmin=pmin,
    pmax=pmax,
    generator_name=wind_generator,
    wind_capacity_factors=gen_capacity_factor,
)

# create a `Tracker` using`mp_wind_battery`
project_tracker_object = Tracker(
    tracking_model_object=mp_wind_battery_track_project,
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
    self_schedule=True,
)

## Prescient requires the following functions in this module
get_configuration = coordinator.get_configuration
register_plugins = coordinator.register_plugins
