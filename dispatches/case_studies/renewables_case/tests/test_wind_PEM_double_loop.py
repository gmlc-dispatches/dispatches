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
import pytest
from pathlib import Path
import pandas as pd
import os
import numpy as np
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from pyomo.common.unittest import assertStructuredAlmostEqual

from idaes.apps.grid_integration.tracker import Tracker
from idaes.apps.grid_integration.bidder import SelfScheduler, Bidder
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.model_data import (
    RenewableGeneratorModelData,
    ThermalGeneratorModelData,
)
from dispatches.case_studies.renewables_case.wind_PEM_double_loop import MultiPeriodWindPEM, update_wind_capacity_factor
from dispatches.case_studies.renewables_case.PEM_parametrized_bidder import PEMParametrizedBidder, PerfectForecaster


re_case_dir = Path(this_file_dir()).parent
wind_thermal_dispatch_data = pd.read_csv(re_case_dir / "data" / "Wind_Thermal_Dispatch.csv")
wind_thermal_dispatch_data.index = pd.to_datetime(wind_thermal_dispatch_data["DateTime"])

tracking_horizon = 4
day_ahead_horizon = 48
real_time_horizon = 4
pmin = 0
pmax = 200
bus_name = "Carter"

generator_params = {
    "gen_name": "309_WIND_1",
    "bus": bus_name,
    "p_min": pmin,
    "p_max": pmax,
    "p_cost": 0,
    "fixed_commitment": None,
}


def test_track_market_dispatch():
    solver = pyo.SolverFactory("ipopt")

    model_data = RenewableGeneratorModelData(**generator_params)

    mp_wind_battery = MultiPeriodWindPEM(
        model_data=model_data,
        wind_capacity_factors=wind_thermal_dispatch_data["309_WIND_1-RTCF"].values,
        wind_pmax_mw=pmax,
        pem_pmax_mw=25
    )

    n_tracking_hour = 1

    # create a `Tracker` using`mp_wind_battery`
    tracker_object = Tracker(
        tracking_model_object=mp_wind_battery,
        tracking_horizon=tracking_horizon,
        n_tracking_hour=n_tracking_hour,
        solver=solver,
    )

    cap_factors = mp_wind_battery._get_capacity_factors(tracker_object.model.fs)
    assert cap_factors[0] == pytest.approx(0.00562, rel=1e-3)

    # example market dispatch signal for 4 hours
    market_dispatch = [0, 1.5, 15.0, 24.5]

    # find a solution that tracks the dispatch signal
    tracker_object.track_market_dispatch(
        market_dispatch=market_dispatch, date="2020-01-02", hour="00:00"
    )

    blks = tracker_object.model.fs.windPEM.get_active_process_blocks()
    assert len(blks) == tracking_horizon

    wind_power = [blks[i].fs.windpower.electricity[0].value for i in range(4)]
    wind_waste = [pyo.value(tracker_object.model.fs.wind_waste[i]) for i in range(4)]
    expected_wind_power = [1123.85, 1573.38, 20510, 25938]

    for power, expected in zip(wind_power, expected_wind_power):
        assert (
            pytest.approx(power, rel=1e-3) == expected
        )
    for power in wind_waste:
        assert (
            pytest.approx(power, abs=1e-3) == 0
        )

    produced_power = [pyo.value(tracker_object.power_output[t])
                      for t in range(tracking_horizon)]

    for power, expected in zip(produced_power, market_dispatch):
        assert (
            pytest.approx(power, abs=1e-3) == expected
        )

    expected_pem_energy = [expected_wind_power[i] -
                               market_dispatch[i] * 1e3 for i in range(tracking_horizon)]
    pem_power = [blks[i].fs.pem.electricity[0].value for i in range(4)]

    for power, expected in zip(pem_power, expected_pem_energy):
        assert (
            pytest.approx(power, rel=1e-3) == expected
        )

    mp_wind_battery.update_model(tracker_object.model.fs, [0] * 4)


def test_compute_parametrized_bids_DA():
    forecaster = PerfectForecaster(wind_thermal_dispatch_data)

    model_data = RenewableGeneratorModelData(**generator_params)

    solver = pyo.SolverFactory("cbc")

    mp_wind_battery_bid = MultiPeriodWindPEM(
        model_data=model_data,
        wind_capacity_factors=wind_thermal_dispatch_data["309_WIND_1-RTCF"].values,
        wind_pmax_mw=pmax,
        pem_pmax_mw=25
    )

    bidder_object = PEMParametrizedBidder(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        solver=solver,
        forecaster=forecaster,
        pem_marginal_cost=30,
        pem_mw=25
    )

    date = "2020-01-02"
    bid_energies = bidder_object.compute_day_ahead_bids(date=date)
    bid_energies = [i['309_WIND_1']['p_max'] for i in bid_energies.values()]

    known_solution = [0.13, 1.08, 3.64, 15.37, 24.68, 31.83, 33.18, 13.89, 7.55, 4.99, 1.08, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.81, 1.89, 33.45, 34.12, 17.8, 13.49, 
                      48.15, 69.72, 36.01, 111.67, 132.7, 178.56, 107.89, 146.86, 145.38, 143.9, 
                      156.84, 111.94, 87.66, 154.69, 79.3, 116.12, 111.8, 98.18, 98.58, 106.54, 
                      130.14, 124.75, 160.08, 153.47, 162.1]

    assertStructuredAlmostEqual(bid_energies, known_solution, abstol=1e-2)


def test_compute_parametrized_bids_RT():
    wind_pmax = 200
    pem_pmax = 25

    forecaster = PerfectForecaster(wind_thermal_dispatch_data)

    thermal_generator_params = {
        "gen_name": "309_WIND_1",
        "bus": bus_name,
        "p_min": pmin,
        "p_max": wind_pmax,
        "min_down_time": 0,
        "min_up_time": 0,
        "ramp_up_60min": wind_pmax + pem_pmax,
        "ramp_down_60min": wind_pmax + pem_pmax,
        "shutdown_capacity": wind_pmax + pem_pmax,
        "startup_capacity": 0,
        "initial_status": 1,
        "initial_p_output": 0,
        "production_cost_bid_pairs": [(pmin, 0), (wind_pmax, 0)],
        "include_default_p_cost": False,
        "startup_cost_pairs": [(0, 0)],
        "fixed_commitment": None,
    }
    model_data = ThermalGeneratorModelData(**thermal_generator_params)

    solver = pyo.SolverFactory("cbc")

    mp_wind_battery_bid = MultiPeriodWindPEM(
        model_data=model_data,
        wind_capacity_factors=wind_thermal_dispatch_data["309_WIND_1-RTCF"].values,
        wind_pmax_mw=wind_pmax,
        pem_pmax_mw=pem_pmax
    )

    bidder_object = PEMParametrizedBidder(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        solver=solver,
        forecaster=forecaster,
        pem_marginal_cost=30,
        pem_mw=25
    )

    date = "2020-01-02"
    bids = bidder_object.compute_real_time_bids(date=date, hour=0, realized_day_ahead_prices=None, realized_day_ahead_dispatches=None)
    bids = [i['309_WIND_1']['p_cost'] for i in bids.values()]
    bid_prices = [bid[-1][1] for bid in bids]

    known_solution = [33.72, 47.2, 615.31, 750.0]

    assertStructuredAlmostEqual(bid_prices, known_solution, reltol=1e-2)

    bidder_object.write_results(this_file_dir())
    os.remove(Path(this_file_dir()) / "bidder_detail.csv")