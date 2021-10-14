#################################################################################
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
#################################################################################
import pytest
# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var, TerminationCondition, SolverStatus
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock

from dispatches.models.renewables_case.battery import BatteryStorage


def test_battery_init():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.battery = BatteryStorage()

    assert hasattr(m.fs.battery, "dt")
    assert isinstance(m.fs.battery.initial_state_of_charge, Var)
    assert isinstance(m.fs.battery.initial_energy_throughput, Var)
    assert isinstance(m.fs.battery.nameplate_power, Var)
    assert isinstance(m.fs.battery.nameplate_energy, Var)
    assert isinstance(m.fs.battery.elec_in, Var)
    assert isinstance(m.fs.battery.elec_out, Var)


def test_battery_solve():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.battery = BatteryStorage()
    m.fs.battery.dt.set_value(1)
    m.fs.battery.nameplate_power.set_value(5)
    m.fs.battery.nameplate_energy.fix(20)

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    m.fs.battery.elec_in.fix(5)
    m.fs.battery.elec_out.fix(0)

    assert_units_consistent(m)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.battery.state_of_charge.value == 5.0
    assert m.fs.battery.energy_throughput.value == 2.5


def test_battery_solve_1():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.battery = BatteryStorage()

    m.fs.battery.nameplate_power.set_value(5)
    m.fs.battery.nameplate_energy.fix(20)
    m.fs.battery.dt.set_value(1)

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    m.fs.battery.elec_in.fix(5)
    m.fs.battery.state_of_charge.fix(5.0)
    m.fs.battery.energy_throughput.fix(2.5)

    assert_units_consistent(m)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.battery.elec_out.value == 0


def test_battery_solve_2():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.battery = BatteryStorage()

    m.fs.battery.nameplate_power.set_value(5)
    m.fs.battery.nameplate_energy.fix(20)
    m.fs.battery.dt.set_value(1)

    m.fs.battery.elec_in.fix(5)
    m.fs.battery.elec_out.fix(0)
    m.fs.battery.state_of_charge.fix(5.0)
    m.fs.battery.energy_throughput.fix(2.5)

    assert_units_consistent(m)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.battery.initial_state_of_charge.value == 0
    assert m.fs.battery.initial_energy_throughput.value == 0
