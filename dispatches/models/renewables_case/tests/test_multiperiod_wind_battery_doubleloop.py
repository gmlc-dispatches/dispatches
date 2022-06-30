import pytest
import pyomo.environ as pyo
from idaes.apps.grid_integration.tracker import Tracker
from idaes.apps.grid_integration.bidder import SelfScheduler
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.model_data import (
    RenewableGeneratorModelData,
    ThermalGeneratorModelData,
)
from dispatches.models.renewables_case.wind_battery_double_loop import MultiPeriodWindBattery
from pyomo.common.unittest import assertStructuredAlmostEqual


gen_capacity_factor = [0.006, 0.008, 0.103, 0.13, 0.175, 0.162, 0.06, 0.03, 0.022, 0.007, 0.007, 0.006, 0.006, 0.005, 0.007, 0.006, 0.012, 0.009, 0.007, 0.024, 0.103, 0.31,
                       0.499, 0.473, 0.22, 0.617, 0.312, 0.362, 0.392, 0.525, 0.56, 0.474, 0.501, 0.516, 0.385, 0.482, 0.415, 0.54, 0.555, 0.439, 0.43, 0.485, 0.615, 0.444, 0.232, 0.215, 0.158, 0.179]
historical_da_prices = [21.3, 20.4, 19.7, 20.0, 20.0, 20.4, 21.8, 23.4, 18.1, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 18.9, 22.5, 33.8, 33.8, 27.1, 24.6,
                        23.1, 19.7, 19.0, 30.9, 26.3, 22.0, 26.3, 30.7, 33.2, 37.7, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 19.4, 37.5, 36.1, 33.8, 26.8, 22.5, 19.7, 18.9]
historical_rt_prices = [23.1, 23.0, 24.6, 24.6, 27.1, 22.5, 22.5, 23.7, 21.8, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 33.2, 36.5, 35.5, 34.4, 33.2,
                        30.7, 18.9, 18.9, 29.6, 30.7, 30.9, 22.5, 22.5, 35.0, 51.4, 33.2, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 20.8, 21.3, 23.6, 23.1, 23.7, 23.2]


@pytest.mark.component
def test_track_market_dispatch():

    tracking_horizon = 4
    pmin = 0
    pmax = 200

    solver = pyo.SolverFactory("ipopt")

    mp_wind_battery = MultiPeriodWindBattery(
        horizon=tracking_horizon,
        pmin=pmin,
        pmax=pmax,
        default_bid_curve=None,
        generator_name="309_WIND_1",
        wind_capacity_factors=gen_capacity_factor,
    )

    n_tracking_hour = 1

    # create a `Tracker` using`mp_wind_battery`
    tracker_object = Tracker(
        tracking_model_object=mp_wind_battery,
        n_tracking_hour=n_tracking_hour,
        solver=solver,
    )

    # example market dispatch signal for 4 hours
    market_dispatch = [0, 3.5, 15.0, 24.5]

    # find a solution that tracks the dispatch signal
    tracker_object.track_market_dispatch(
        market_dispatch=market_dispatch, date="2020-01-02", hour="00:00"
    )

    blks = tracker_object.model.fs.windBattery.get_active_process_blocks()
    assert len(blks) == tracking_horizon

    for t, dispatch in zip(range(tracking_horizon), market_dispatch):
        assert (
            pytest.approx(pyo.value(tracker_object.power_output[t]), abs=1e-3)
            == dispatch
        )

    last_delivered_power = market_dispatch[0]
    assert (
        pytest.approx(tracker_object.get_last_delivered_power(), abs=1e-3)
        == last_delivered_power
    )


@pytest.mark.component
def test_compute_bids_self_schedule():
    day_ahead_horizon = 48
    real_time_horizon = 4
    n_scenario = 1
    pmin = 0
    pmax = 200
    bus_name = "Carter"

    backcaster = Backcaster({bus_name: historical_da_prices}, {bus_name: historical_rt_prices})

    generator_params = {
        "gen_name": "309_WIND_1",
        "bus": bus_name,
        "p_min": pmin,
        "p_max": pmax,
        "p_cost": 0,
        "fixed_commitment": None,
    }
    model_data = RenewableGeneratorModelData(**generator_params)

    solver = pyo.SolverFactory("cbc")

    mp_wind_battery_bid = MultiPeriodWindBattery(
        model_data=model_data,
        wind_capacity_factors=gen_capacity_factor,
        wind_pmax_mw=pmax,
        battery_pmax_mw=25,
        battery_energy_capacity_mwh=100,
    )

    bidder_object = SelfScheduler(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        n_scenario=n_scenario,
        solver=solver,
        forecaster=backcaster,
    )

    date = "2020-01-02"
    bids = bidder_object.compute_bids(date=date)

    blks = bidder_object.model.fs[0].windBattery.get_active_process_blocks()
    assert len(blks) == day_ahead_horizon
    assert len(bidder_object.model.fs.index_set()) == n_scenario

    # test against known solution with ipopt
    known_solution = {
        "309_WIND_1": [
            0.0,
            0.0,
            0.0,
            0.0,
            26.60674608336997,
            33.17599460552932,
            28.324009263281173,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            26.888064733648,
            58.44571813890762,
            59.111726109855844,
            42.80175320296694,
            13.48617666891436,
            0.0,
            0.0,
            61.00809170600134,
            86.66554281861092,
            107.70397842211733,
            178.55697909642618,
            132.88941335131491,
            171.8644639244774,
            170.38098449089685,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            98.58395144976396,
            131.54079568442347,
            155.14160485502362,
            149.696734240835,
            185.0809170600135,
            153.47269049224542,
            0.0,
            0.0,
        ]
    }

    assertStructuredAlmostEqual(bids, known_solution)

test_compute_bids_self_schedule()