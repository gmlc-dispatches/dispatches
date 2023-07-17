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
import platform
from idaes.core.util.model_statistics import degrees_of_freedom

from dispatches.case_studies.renewables_case.RE_flowsheet import *
from dispatches.case_studies.renewables_case.wind_battery_LMP import wind_battery_optimize, record_results, plot_results
from dispatches.case_studies.renewables_case.wind_battery_PEM_LMP import wind_battery_pem_optimize
from dispatches.case_studies.renewables_case.wind_battery_PEM_tank_turbine_LMP import wind_battery_pem_tank_turb_optimize

@pytest.fixture
def input_params():
    params = copy.copy(default_input_params)
    with open(re_case_dir / 'tests' / 'rts_results_all_prices.npy', 'rb') as f:
        _ = np.load(f)
        price = np.load(f)

        prices_used = copy.copy(price)
        prices_used[prices_used > 200] = 200
    params['DA_LMPs'] = prices_used

    # wind resource data from example Wind Toolkit file
    wind_data = SRW_to_wind_data(re_case_dir / 'data' / '44.21_-101.94_windtoolkit_2012_60min_80m.srw')
    wind_speeds = [wind_data['data'][i][2] for i in range(8760)]
    wind_resource = {t:
                        {'wind_resource_config': {
                            'resource_speed': [wind_speeds[t]]
                        }
                    } for t in range(8760)}
    params["wind_resource"] = wind_resource
    return params


def test_create_model(input_params):
    tank_type = "simple"
    m = create_model(
        re_mw=fixed_wind_mw,
        pem_bar=pem_bar,
        batt_mw=fixed_batt_mw,
        tank_type=tank_type,
        tank_length_m=fixed_tank_size,
        turb_inlet_bar=pem_bar,
        resource_config=input_params['wind_resource'][0]['wind_resource_config']
    )

    assert hasattr(m.fs, "windpower")
    assert hasattr(m.fs, "splitter")
    assert hasattr(m.fs, "battery")
    assert hasattr(m.fs, "pem")
    assert hasattr(m.fs, "h2_tank")
    assert hasattr(m.fs, "translator")
    if tank_type != "simple":
        assert hasattr(m.fs, "h2_splitter")
        assert hasattr(m.fs, "tank_sold")
    assert hasattr(m.fs, "mixer")
    assert hasattr(m.fs, "h2_turbine")
    assert hasattr(m.fs.mixer, "purchased_hydrogen_feed")

    assert m.fs.windpower.system_capacity.fixed
    assert m.fs.battery.nameplate_power.fixed
    if tank_type != "simple":
        assert m.fs.h2_tank.tank_length[0].fixed
        assert m.fs.h2_tank.energy_balances.active
    assert m.fs.mixer.air_h2_ratio.active
    assert m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].lb
    assert value(m.fs.h2_turbine.turbine.deltaP[0]) == -2401000.0
    assert value(m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"]) == 2e-4

    dof = degrees_of_freedom(m)
    assert dof == 10


def test_create_model_PV():
    pv_capacity_factors = {'capacity_factor': [0.5]}

    tank_type = "simple"
    m = create_model(
        re_mw=800,
        pem_bar=pem_bar,
        batt_mw=fixed_batt_mw,
        tank_type=tank_type,
        tank_length_m=fixed_tank_size,
        turb_inlet_bar=pem_bar,
        resource_config=pv_capacity_factors,
        re_type='pv'
    )

    assert hasattr(m.fs, "pv")
    assert hasattr(m.fs, "splitter")
    assert hasattr(m.fs, "battery")
    assert hasattr(m.fs, "pem")
    assert hasattr(m.fs, "h2_tank")
    assert hasattr(m.fs, "translator")
    assert hasattr(m.fs, "mixer")
    assert hasattr(m.fs, "h2_turbine")
    assert hasattr(m.fs.mixer, "purchased_hydrogen_feed")

    assert m.fs.pv.system_capacity.fixed
    assert m.fs.battery.nameplate_power.fixed
    assert m.fs.mixer.air_h2_ratio.active
    assert m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].lb
    assert value(m.fs.h2_turbine.turbine.deltaP[0]) == -2401000.0
    assert value(m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"]) == 2e-4

    dof = degrees_of_freedom(m)
    assert dof == 10


def test_wind_battery_optimize(input_params):
    mp = wind_battery_optimize(n_time_points=7 * 24, input_params=input_params, verbose=True)
    assert value(mp.pyomo_model.NPV) == pytest.approx(666049365, rel=1e-3)
    assert value(mp.pyomo_model.annual_revenue) == pytest.approx(59163455, rel=1e-3)
    blks = mp.get_active_process_blocks()
    assert value(blks[0].fs.battery.nameplate_power) == pytest.approx(0, abs=1)
    plot_results(*record_results(mp))


def test_wind_pem_optimize(input_params):
    input_params['h2_price_per_kg'] = 2.5
    input_params['design_opt'] = "PEM"
    input_params['batt_mw'] = 0
    design_res, _ = wind_battery_pem_optimize(time_points=6 * 24, input_params=input_params, verbose=True)
    assert design_res['batt_mw'] == pytest.approx(0, rel=1e-3)
    assert design_res['pem_mw'] == pytest.approx(487, rel=1e-2)
    assert design_res['annual_rev_h2'] == pytest.approx(155129116, rel=1e-2)
    assert design_res['annual_rev_E'] == pytest.approx(68599396, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(1339462317, rel=1e-2)


def test_wind_battery_pem_optimize(input_params):
    input_params['h2_price_per_kg'] = 2.5
    design_res, _ = wind_battery_pem_optimize(time_points=6 * 24, input_params=input_params, verbose=True)
    assert design_res['batt_mw'] == pytest.approx(0, rel=1e-3)
    assert design_res['pem_mw'] == pytest.approx(487, abs=1)
    assert design_res['annual_rev_h2'] == pytest.approx(155129116, rel=1e-2)
    assert design_res['annual_rev_E'] == pytest.approx(68599396, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(1339462317, rel=1e-2)


def test_wind_battery_pem_tank_turb_optimize_simple(input_params):
    input_params['h2_price_per_kg'] = 2.0
    design_res = wind_battery_pem_tank_turb_optimize(6 * 24, input_params, verbose=True, plot=False)
    assert design_res['NPV'] == pytest.approx(1018975372, rel=1e-2)
    assert design_res['batt_mw'] == pytest.approx(0, abs=3)
    assert design_res['pem_mw'] == pytest.approx(355, abs=3)
    assert design_res['tank_kgH2'] == pytest.approx(0, abs=3)
    assert design_res['turb_mw'] == pytest.approx(0, abs=3)
    assert design_res['avg_turb_eff'] == pytest.approx(1.51, rel=1e-1)
    assert design_res['annual_rev_h2'] == pytest.approx(99396474, abs=5e3)
    assert design_res['annual_rev_E'] == pytest.approx(28711076, rel=1e-2)


@pytest.mark.skipif(platform.system() == "Windows", reason="Platform differences in IPOPT solve")
def test_wind_battery_pem_tank_turb_optimize_detailed(input_params):
    input_params['h2_price_per_kg'] = 2.0
    input_params['tank_type'] = 'detailed'
    input_params['pem_mw'] = 0
    design_res = wind_battery_pem_tank_turb_optimize(6 * 24, input_params=input_params, verbose=True, plot=False)
    assert design_res['batt_mw'] == pytest.approx(0, abs=3)
    assert design_res['pem_mw'] == pytest.approx(355, abs=3)
    assert design_res['tank_kgH2'] == pytest.approx(0, abs=3)
    assert design_res['turb_mw'] == pytest.approx(0, abs=3)
    assert design_res['avg_turb_eff'] == pytest.approx(1.50, rel=1e-1)
    assert design_res['annual_rev_h2'] == pytest.approx(99396474, abs=5e3)
    assert design_res['annual_rev_E'] == pytest.approx(28711076, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(1018975372, rel=1e-2)
