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
import platform
from idaes.core.util.model_statistics import degrees_of_freedom

from dispatches.models.nuclear_case.unit_models.hydrogen_tank import HydrogenTank as DetailedHydrogenTank
from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.wind_battery_LMP import wind_battery_optimize, record_results, plot_results
from dispatches.models.renewables_case.wind_battery_PEM_LMP import wind_battery_pem_optimize
from dispatches.models.renewables_case.wind_battery_PEM_tank_turbine_LMP import wind_battery_pem_tank_turb_optimize

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

def test_h2_valve_opening():
    valve_coef = 0.03380
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    m.fs.h2_tank = DetailedHydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})
    m.fs.h2_tank.tank_diameter.fix(0.1)
    m.fs.h2_tank.tank_length.fix(0.3)
    m.fs.h2_tank.control_volume.properties_in[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.h2_tank.control_volume.properties_out[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.h2_tank.previous_state[0].pressure.setub(max_pressure_bar * 1e5)
    # hydrogen tank valve
    m.fs.tank_valve = Valve(
        default={
            "valve_function_callback": ValveFunctionType.linear,
            "property_package": m.fs.h2ideal_props,
        }
    )
    # connect tank to the valve
    m.fs.tank_to_valve = Arc(
        source=m.fs.h2_tank.outlet,
        destination=m.fs.tank_valve.inlet
    )
    m.fs.tank_valve.outlet.pressure[0].fix(3 * 1e5)
    # NS: tuning valve's coefficient of flow to match the condition
    m.fs.tank_valve.Cv.fix(valve_coef)
    # NS: unfixing valve opening. This allows for controlling both pressure
    # and flow at the outlet of the valve
    m.fs.tank_valve.valve_opening[0].unfix()
    m.fs.tank_valve.valve_opening[0].setlb(0)
    m.fs.h2_tank.dt[0].fix(timestep_hrs * 3600)

    assert m.fs.h2_tank.tank_diameter[0].fixed
    assert m.fs.h2_tank.tank_length[0].fixed
    assert m.fs.tank_valve.outlet.pressure[0].fixed

    m.fs.h2_tank.inlet.flow_mol[0].fix(9992.784)
    m.fs.h2_tank.outlet.flow_mol[0].fix(4251.928)

    m.fs.tank_valve.valve_opening[0].setlb(0)
    m.fs.tank_valve.valve_opening[0].setub(1)

    m.fs.h2_tank.previous_material_holdup[(0.0, 'Vap', 'hydrogen')].fix(17.832)
    dof = degrees_of_freedom(m)
    assert dof == 8

    opt = SolverFactory("ipopt")
    res = opt.solve(m, tee=True)

    assert res.Solver.status == 'ok'
    assert value(m.fs.h2_tank.material_holdup[(0.0, 'Vap', 'hydrogen')]) == pytest.approx(22.3761, rel=1e-3)
    assert value(m.fs.tank_valve.valve_opening[0]) == pytest.approx(0.83331, rel=1e-3)


def test_create_model(input_params):
    tank_type = "simple"
    m = create_model(
        wind_mw=fixed_wind_mw,
        pem_bar=pem_bar,
        batt_mw=fixed_batt_mw,
        tank_type=tank_type,
        tank_length_m=fixed_tank_size,
        turb_inlet_bar=pem_bar,
        wind_resource_config=input_params['wind_resource'][0]['wind_resource_config']
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


def test_wind_battery_optimize(input_params):
    mp = wind_battery_optimize(n_time_points=7 * 24, input_params=input_params, verbose=True)
    assert value(mp.pyomo_model.NPV) == pytest.approx(1001068228, rel=1e-3)
    assert value(mp.pyomo_model.annual_revenue) == pytest.approx(168691601, rel=1e-3)
    blks = mp.get_active_process_blocks()
    assert value(blks[0].fs.battery.nameplate_power) == pytest.approx(1326779, rel=1e-3)
    plot_results(*record_results(mp))


def test_wind_battery_pem_optimize(input_params):
    input_params['h2_price_per_kg'] = 2.5
    design_res, _ = wind_battery_pem_optimize(time_points=6 * 24, input_params=input_params, verbose=True)
    assert design_res['batt_mw'] == pytest.approx(4874, rel=1e-3)
    assert design_res['pem_mw'] == pytest.approx(0, abs=1e-1)
    assert design_res['annual_rev_h2'] == pytest.approx(0.055, abs=1e-1)
    assert design_res['annual_rev_E'] == pytest.approx(531576401, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(2322131921, rel=1e-2)


def test_wind_battery_pem_tank_turb_optimize_simple(input_params):
    input_params['h2_price_per_kg'] = 2.0
    design_res = wind_battery_pem_tank_turb_optimize(6 * 24, input_params, verbose=False, plot=False)
    assert design_res['batt_mw'] == pytest.approx(4874, rel=1e-2)
    assert design_res['pem_mw'] == pytest.approx(0, abs=3)
    assert design_res['tank_kgH2'] == pytest.approx(0, abs=3)
    assert design_res['turb_mw'] == pytest.approx(0, abs=3)
    assert design_res['avg_turb_eff'] == pytest.approx(1.51, rel=1e-1)
    assert design_res['annual_rev_h2'] == pytest.approx(2634, abs=5e3)
    assert design_res['annual_rev_E'] == pytest.approx(531566543, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(2344545889, rel=1e-2)


@pytest.mark.skipif(platform.system() == "Windows", reason="Platform differences in IPOPT solve")
def test_wind_battery_pem_tank_turb_optimize_detailed(input_params):
    input_params['h2_price_per_kg'] = 2.0
    input_params['tank_type'] = 'detailed'
    design_res = wind_battery_pem_tank_turb_optimize(6 * 24, input_params=input_params, verbose=True, plot=False)
    assert design_res['batt_mw'] == pytest.approx(4874, rel=1e-2)
    assert design_res['pem_mw'] == pytest.approx(0, abs=3)
    assert design_res['tank_kgH2'] == pytest.approx(0, abs=3)
    assert design_res['turb_mw'] == pytest.approx(0, abs=3)
    assert design_res['avg_turb_eff'] == pytest.approx(1.51, rel=1e-1)
    assert design_res['annual_rev_h2'] == pytest.approx(2634, abs=5e3)
    assert design_res['annual_rev_E'] == pytest.approx(531566543, rel=1e-2)
    assert design_res['NPV'] == pytest.approx(2344545889, rel=1e-2)
