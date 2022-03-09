##############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
##############################################################################
import pytest
import itertools

from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.wind_LMP import wind_optimize
from dispatches.models.renewables_case.wind_battery_LMP import wind_battery_optimize
from dispatches.models.renewables_case.wind_PEM_LMP import wind_pem_optimize
from dispatches.models.renewables_case.wind_PEM_tank_LMP import wind_pem_tank_optimize
from dispatches.models.renewables_case.wind_battery_PEM_LMP import wind_battery_pem_optimize
from dispatches.models.renewables_case.wind_battery_PEM_tank_LMP import wind_battery_pem_tank_optimize
from dispatches.models.renewables_case.turbine_LMP import turb_optimize
from dispatches.models.renewables_case.wind_battery_PEM_tank_turbine_LMP import wind_battery_pem_tank_turb_optimize


def test_h2_valve_opening():
    valve_coef = 0.03380
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    add_h2_tank(m, 3, 0.3, valve_coef)

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


def test_create_model():
    m = create_model(
        wind_mw=fixed_wind_mw,
        pem_bar=pem_bar,
        batt_mw=fixed_batt_mw,
        valve_cv=valve_cv,
        tank_len_m=fixed_tank_size,
        h2_turb_bar=pem_bar,
        wind_resource_config=wind_resource[0]['wind_resource_config'],
        verbose=False,
    )

    assert hasattr(m.fs, "windpower")
    assert hasattr(m.fs, "splitter")
    assert hasattr(m.fs, "battery")
    assert hasattr(m.fs, "pem")
    assert hasattr(m.fs, "h2_tank")
    assert hasattr(m.fs, "translator")
    assert hasattr(m.fs, "h2_splitter")
    assert hasattr(m.fs, "tank_sold")
    assert hasattr(m.fs, "mixer")
    assert hasattr(m.fs, "h2_turbine")
    assert hasattr(m.fs.mixer, "purchased_hydrogen_feed")

    assert m.fs.windpower.system_capacity.fixed
    assert m.fs.battery.nameplate_power.fixed
    assert m.fs.h2_tank.tank_length[0].fixed
    assert m.fs.h2_tank.energy_balances.active
    assert m.fs.mixer.air_h2_ratio.active
    assert m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].lb
    assert value(m.fs.h2_turbine.turbine.deltaP[0]) == -2401000.0
    assert value(m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"]) == 2e-4

    dof = degrees_of_freedom(m)
    assert dof == 10

    opt = SolverFactory("ipopt")
    res = opt.solve(m)

    assert res.Solver.status == 'ok'


def test_wind_optimize():
    wind_cap, profit, npv = wind_optimize(n_time_points=7 * 24, h2_price=2, verbose=False)
    print(wind_cap, profit, npv)
    assert wind_cap == pytest.approx(200)
    assert profit == pytest.approx(263976, rel=1e-3)
    assert npv == pytest.approx(-99985625, rel=1e-3)


def test_turbine_optimize():
    turb_cap, turb_eff, h2_revenue, elec_revenue, NPV = turb_optimize(n_time_points=24, h2_price=2, pem_pres_bar=pem_bar, turb_op_bar=25.2, verbose=False)
    assert turb_eff == pytest.approx(1.462, rel=1e-3)
    assert h2_revenue == pytest.approx(-0.3456, rel=1e-3)
    assert elec_revenue == pytest.approx(-31.8937, rel=1e-3)
    assert NPV == pytest.approx(-1025770, rel=1e-3) 

    turb_cap, turb_eff, h2_revenue, elec_revenue, NPV = turb_optimize(n_time_points=24, h2_price=2, pem_pres_bar=pem_bar, turb_op_bar=30, verbose=False)
    assert turb_eff == pytest.approx(1.436, rel=1e-3)
    assert h2_revenue == pytest.approx(-0.3456, rel=1e-3)
    assert elec_revenue == pytest.approx(-31.8937, rel=1e-3)
    assert NPV == pytest.approx(-1025770, rel=1e-3) 


def test_wind_battery_optimize():
    mp = wind_battery_optimize(n_time_points=7 * 24, verbose=False)
    assert value(mp.pyomo_model.NPV) == pytest.approx(334669831, rel=1e-3)
    assert value(mp.pyomo_model.annual_revenue) == pytest.approx(53744366, rel=1e-3)
    blks = mp.get_active_process_blocks()
    assert value(blks[0].fs.battery.nameplate_power) == pytest.approx(409594, rel=1e-3)


def test_wind_pem_optimize():
    wind_cap, pem_cap, h2_revenue, elec_revenue, NPV = wind_pem_optimize(n_time_points=7 * 24, h2_price=2, verbose=False)
    assert pem_cap == pytest.approx(24.619, rel=1e-3)
    assert h2_revenue == pytest.approx(125817, rel=1e-3)
    assert elec_revenue == pytest.approx(194901, rel=1e-3)
    assert NPV == pytest.approx(216242601, rel=1e-3)


def test_wind_pem_tank_optimize():
    wind_cap, pem_cap, tank_size, h2_revenue, elec_revenue, NPV = wind_pem_tank_optimize(n_time_points=7 * 24, h2_price=2, verbose=False)
    assert pem_cap == pytest.approx(24.619, rel=1e-3)
    assert tank_size == pytest.approx(0, abs=1e-3)
    assert h2_revenue == pytest.approx(125817, rel=1e-3)
    assert elec_revenue == pytest.approx(194901, rel=1e-3)
    assert NPV == pytest.approx(216242601, rel=1e-3)


def test_wind_battery_pem_optimize():
    design_res, _, __ = wind_battery_pem_optimize(time_points=7 * 24, h2_price=2.5, verbose=False)
    assert design_res['batt_mw'] == pytest.approx(361.191, rel=1e-3)
    assert design_res['pem_mw'] == pytest.approx(13.481, rel=1e-3)
    assert design_res['annual_rev_h2'] == pytest.approx(4631234, rel=1e-3)
    assert design_res['annual_rev_E'] == pytest.approx(46886638, rel=1e-3)
    assert design_res['NPV'] == pytest.approx(338729537, rel=1e-3)

def test_wind_battery_pem_tank_optimize():
    wind_cap, batt_cap, pem_cap, tank_size, h2_revenue, elec_revenue, NPV = wind_battery_pem_tank_optimize(n_time_points=6 * 24, h2_price=2.5, verbose=False)
    assert batt_cap == pytest.approx(351.103, rel=1e-3)
    assert pem_cap == pytest.approx(13.481, rel=1e-3)
    assert tank_size == pytest.approx(0, abs=1e-3)
    assert h2_revenue == pytest.approx(81416, rel=1e-3)
    assert elec_revenue == pytest.approx(904939, rel=1e-3)
    assert NPV == pytest.approx(345163267, rel=1e-3)


def test_wind_battery_pem_tank_turb_optimize():
    design_res, _, __ = wind_battery_pem_tank_turb_optimize(time_points=6 * 24, h2_price=2.5, verbose=False)
    assert design_res['batt_mw'] == pytest.approx(1000, rel=1e-3)
    assert design_res['pem_mw'] == pytest.approx(0.0001033, rel=1e-3)
    assert design_res['tank_kgH2'] == pytest.approx(0.002243, rel=1e-3)
    assert design_res['turb_mw'] == pytest.approx(0, abs=1e-3)
    assert design_res['avg_turb_eff'] == pytest.approx(1.5139, abs=1e-3)
    assert design_res['annual_rev_h2'] == pytest.approx(-40.830, rel=1e-3)
    assert design_res['annual_rev_E'] == pytest.approx(115145468, rel=1e-3)
    assert design_res['NPV'] == pytest.approx(574934934, rel=1e-3)
