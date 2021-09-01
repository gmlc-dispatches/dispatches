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
    opening = 0.0001
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    h2_tank, tank_valve = add_h2_tank(m, 3, 0.3, opening)

    h2_tank.inlet.pressure.fix(8e6)
    h2_tank.inlet.temperature.fix(300)
    h2_tank.inlet.flow_mol[0] = 0
    tank_valve.outlet.pressure.fix(8e6)


def test_model():
    battery_discharge_kw = [-1.9051, 0, -3.81025]
    h2_out_mol_per_s = [0.001, 0.003, 0.002]

    ok, m = run_model(wind_mw=20, pem_bar=3, batt_mw=10, tank_len_m=0.3,
                      battery_discharge_kw=battery_discharge_kw, h2_out_mol_per_s=h2_out_mol_per_s)
    assert ok

    assert value(m.fs.windpower.electricity[0]) == pytest.approx(3.81, 1e-2)
    assert value(m.fs.battery.elec_in[0]) == pytest.approx(3.81, 1e-2)
    assert value(m.fs.battery.state_of_charge[0]) == pytest.approx(5.72, 1e-2)
    assert value(m.fs.splitter.pem_elec[0]) == pytest.approx(0.0, abs=1e-2)
    assert value(m.fs.h2_tank.inlet.flow_mol[0]) == pytest.approx(0.0, abs=1e-2)
    assert value(m.fs.h2_tank.outlet.flow_mol[0]) == 0.002
    assert value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]) == pytest.approx(30.69, 1e-2)
    assert value(m.fs.h2_turbine.turbine.work_mechanical[0] + m.fs.h2_turbine.compressor.work_mechanical[0]) \
           == pytest.approx(-397.65, 1e-2)


def test_model_1():
    battery_discharge_kw = [-1.9051, 0, -3.81025]
    h2_out_mol_per_s = [0.001, 0.003, 0.002]
    ok, m = run_model(wind_mw=200, pem_bar=3, batt_mw=10, tank_len_m=0.3,
                      battery_discharge_kw=battery_discharge_kw, h2_out_mol_per_s=h2_out_mol_per_s)
    assert ok

    assert value(m.fs.windpower.electricity[0]) == pytest.approx(38.1, 1e-2)
    assert value(m.fs.battery.elec_in[0]) == pytest.approx(3.81, 1e-2)
    assert value(m.fs.battery.state_of_charge[0]) == pytest.approx(5.72, 1e-2)
    assert value(m.fs.splitter.pem_elec[0]) == pytest.approx(34.29, abs=1e-2)
    assert value(m.fs.h2_tank.inlet.flow_mol[0]) == pytest.approx(0.087, abs=1e-2)
    assert value(m.fs.h2_tank.outlet.flow_mol[0]) == 0.002
    assert value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]) == pytest.approx(966.73, 1e-2)
    assert value(m.fs.h2_turbine.turbine.work_mechanical[0] + m.fs.h2_turbine.compressor.work_mechanical[0]) \
           == pytest.approx(-397.62, 1e-2)


def test_model_solves_over_range():
    battery_discharge_kw = [-1.9051, 0, -3.81025]
    h2_out_mol_per_s = [0.001, 0.003, 0.002]

    # if the above values are modified given wind and battery size, could do a larger range than what's below
    wind_nameplate_mw = (40, 400)
    PEM_outlet_pressure_bar = (3, 18)
    battery_nameplate_mw = (10, 111)
    H2_tank_length_cm = range(3, 10, 3)

    for w, p, b, t in itertools.product(wind_nameplate_mw, PEM_outlet_pressure_bar, battery_nameplate_mw,
                                        H2_tank_length_cm):
        ok = run_model(wind_mw=w, pem_bar=p, batt_mw=b, tank_len_m=t / 10,
                         battery_discharge_kw=battery_discharge_kw, h2_out_mol_per_s=h2_out_mol_per_s)[0]
        if not ok:
            print(w, p, b, t)
        assert ok
