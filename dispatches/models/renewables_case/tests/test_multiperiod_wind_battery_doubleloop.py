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
