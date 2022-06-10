import pytest
import pyomo.environ as pyo
from idaes.apps.grid_integration.tracker import Tracker
from idaes.apps.grid_integration.bidder import SelfScheduler
from dispatches.models.renewables_case.wind_battery_double_loop import (
    MultiPeriodWindBattery,
    SimpleForecaster,
    gen_capacity_factor,
)
from pyomo.common.unittest import assertStructuredAlmostEqual


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
def test_compute_bids():

    bidding_horizon = 48
    n_scenario = 1
    pmin = 0
    pmax = 200

    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"] = 2

    mp_wind_battery_bid = MultiPeriodWindBattery(
        horizon=bidding_horizon,
        pmin=pmin,
        pmax=pmax,
        generator_name="309_WIND_1",
        wind_capacity_factors=gen_capacity_factor,
    )

    bidder_object = SelfScheduler(
        bidding_model_object=mp_wind_battery_bid,
        n_scenario=n_scenario,
        horizon=bidding_horizon,
        solver=solver,
        forecaster=SimpleForecaster(horizon=bidding_horizon, n_sample=n_scenario),
    )

    date = "2020-01-02"
    bids = bidder_object.compute_bids(date=date)

    blks = bidder_object.model.fs[0].windBattery.get_active_process_blocks()
    assert len(blks) == bidding_horizon
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
