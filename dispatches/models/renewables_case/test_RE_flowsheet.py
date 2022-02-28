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

from .RE_flowsheet import *


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