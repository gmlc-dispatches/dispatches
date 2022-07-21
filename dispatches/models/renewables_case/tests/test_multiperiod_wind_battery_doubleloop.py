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
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
from pyomo.common.unittest import assertStructuredAlmostEqual
from pyomo.common.dependencies import check_min_version

idaes = pytest.importorskip("idaes", minversion="2.0.0.a4", reason="requires at least idaes-pse version 2.0.0.a4")

from idaes.apps.grid_integration.tracker import Tracker
from idaes.apps.grid_integration.bidder import SelfScheduler, Bidder
from idaes.apps.grid_integration.forecaster import Backcaster
from idaes.apps.grid_integration.model_data import (
    RenewableGeneratorModelData,
    ThermalGeneratorModelData,
)
from dispatches.models.renewables_case.wind_battery_double_loop import MultiPeriodWindBattery


@pytest.fixture
def wind_thermal_dispatch_data():
    re_case_dir = Path(this_file_dir()).parent
    df = pd.read_csv(re_case_dir / "data" / "Wind_Thermal_Dispatch.csv")
    df["DateTime"] = df['Unnamed: 0']
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df.index = pd.to_datetime(df["DateTime"])
    return df


@pytest.mark.component
def test_track_market_dispatch(wind_thermal_dispatch_data):

    tracking_horizon = 4
    pmin = 0
    pmax = 200

    solver = pyo.SolverFactory("ipopt")

    generator_params = {
        "gen_name": "309_WIND_1",
        "bus": "Carter",
        "p_min": pmin,
        "p_max": pmax,
        "p_cost": 0,
        "fixed_commitment": None,
    }
    model_data = RenewableGeneratorModelData(**generator_params)

    mp_wind_battery = MultiPeriodWindBattery(
        model_data=model_data,
        wind_capacity_factors=wind_thermal_dispatch_data["309_WIND_1-RTCF"].values,
        wind_pmax_mw=pmax,
        battery_pmax_mw=25,
        battery_energy_capacity_mwh=100,
    )

    n_tracking_hour = 1

    # create a `Tracker` using`mp_wind_battery`
    tracker_object = Tracker(
        tracking_model_object=mp_wind_battery,
        tracking_horizon=tracking_horizon,
        n_tracking_hour=n_tracking_hour,
        solver=solver,
    )

    # example market dispatch signal for 4 hours
    market_dispatch = [0, 1.5, 15.0, 24.5]

    # find a solution that tracks the dispatch signal
    tracker_object.track_market_dispatch(
        market_dispatch=market_dispatch, date="2020-01-02", hour="00:00"
    )

    blks = tracker_object.model.fs.windBattery.get_active_process_blocks()
    assert len(blks) == tracking_horizon

    wind_power = [blks[i].fs.windpower.electricity[0].value for i in range(4)]
    expected_wind_power = [1123.8, 1573.4, 20510.2, 25938.4]

    for power, expected in zip(wind_power, expected_wind_power):
        assert (
            pytest.approx(power, rel=1e-3) == expected
        )

    produced_power = [pyo.value(tracker_object.power_output[t])
                      for t in range(tracking_horizon)]

    for power, expected in zip(produced_power, market_dispatch):
        assert (
            pytest.approx(power, abs=1e-3) == expected
        )

    expected_battery_energy = [expected_wind_power[i] -
                               market_dispatch[i] * 1e3 for i in range(tracking_horizon)]
    battery_power = [blks[i].fs.battery.elec_in[0].value for i in range(4)]

    for power, expected in zip(battery_power, expected_battery_energy):
        assert (
            pytest.approx(power, rel=1e-3) == expected
        )


def test_compute_bids_self_schedule(wind_thermal_dispatch_data):
    day_ahead_horizon = 48
    real_time_horizon = 4
    n_scenario = 1
    pmin = 0
    pmax = 200
    bus_name = "Carter"

    historical_da_prices = wind_thermal_dispatch_data["309_DALMP"].values[0:48].tolist(
    )
    historical_rt_prices = wind_thermal_dispatch_data["309_RTLMP"].values[0:48].tolist(
    )

    backcaster = Backcaster({bus_name: historical_da_prices}, {
                            bus_name: historical_rt_prices})

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
        wind_capacity_factors=wind_thermal_dispatch_data["309_WIND_1-RTCF"].values,
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
    bids = bidder_object.compute_day_ahead_bids(date=date)
    bids = [i['309_WIND_1']['p_max'] for i in bids.values()]

    blks = bidder_object.day_ahead_model.fs[0].windBattery.get_active_process_blocks(
    )
    assert len(blks) == day_ahead_horizon
    assert len(bidder_object.day_ahead_model.fs.index_set()) == n_scenario

    known_solution = [
        0.0, 1.5734, 0.0, 0.0, 10.0865, 32.3219, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.9699, 1.3711, 4.7876, 20.5439, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.0643, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.7721
    ]

    assertStructuredAlmostEqual(bids, known_solution, reltol=1e-2)


def test_compute_bids_thermal_gen(wind_thermal_dispatch_data):
    day_ahead_horizon = 48
    real_time_horizon = 4
    n_scenario = 1
    pmin = 0
    wind_pmax = 200
    battery_pmax = 25
    bus_name = "Carter"

    historical_da_prices = wind_thermal_dispatch_data["309_DALMP"].values[0:48].tolist(
    )
    historical_rt_prices = wind_thermal_dispatch_data["309_RTLMP"].values[0:48].tolist(
    )

    backcaster = Backcaster({bus_name: historical_da_prices}, {
                            bus_name: historical_rt_prices})

    thermal_generator_params = {
        "gen_name": "309_WIND_1",
        "bus": bus_name,
        "p_min": pmin,
        "p_max": wind_pmax,
        "min_down_time": 0,
        "min_up_time": 0,
        "ramp_up_60min": wind_pmax + battery_pmax,
        "ramp_down_60min": wind_pmax + battery_pmax,
        "shutdown_capacity": wind_pmax + battery_pmax,
        "startup_capacity": 0,
        "initial_status": 1,
        "initial_p_output": 0,
        "production_cost_bid_pairs": [(pmin, 0), (wind_pmax, 0)],
        "startup_cost_pairs": [(0, 0)],
        "fixed_commitment": None,
    }
    model_data = ThermalGeneratorModelData(**thermal_generator_params)

    solver = pyo.SolverFactory("cbc")

    mp_wind_battery_bid = MultiPeriodWindBattery(
        model_data=model_data,
        wind_capacity_factors=wind_thermal_dispatch_data["309_WIND_1-RTCF"].values,
        wind_pmax_mw=wind_pmax,
        battery_pmax_mw=battery_pmax,
        battery_energy_capacity_mwh=battery_pmax * 4,
    )

    bidder_object = Bidder(
        bidding_model_object=mp_wind_battery_bid,
        day_ahead_horizon=day_ahead_horizon,
        real_time_horizon=real_time_horizon,
        n_scenario=n_scenario,
        solver=solver,
        forecaster=backcaster,
    )

    date = "2020-01-02"
    bids = bidder_object.compute_day_ahead_bids(date=date)
    bids = [i['309_WIND_1']['p_max'] for i in bids.values()]
    print(bids)

    blks = bidder_object.day_ahead_model.fs[0].windBattery.get_active_process_blocks(
    )
    assert len(blks) == day_ahead_horizon
    assert len(bidder_object.day_ahead_model.fs.index_set()) == n_scenario

    known_solution = [
        0.0, 1.5734, 0.0, 0.0, 10.0865, 32.3219, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 11.9699, 1.3711, 4.7876, 20.5439, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 86.0643, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 35.7721
    ]

    assertStructuredAlmostEqual(bids, known_solution, reltol=1e-2)
