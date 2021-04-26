import pytest
# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock

from dispatches.models.renewables_case.battery import BatteryStorage


def test_battery():
    # Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})   # dynamic or ss flowsheet needs to be specified here

    m.fs.battery = BatteryStorage()
    assert hasattr(m.fs.battery, "dt")
    assert isinstance(m.fs.battery.initial_state_of_charge, Var)
    assert isinstance(m.fs.battery.initial_energy_throughput, Var)
    assert isinstance(m.fs.battery.nameplate_power, Var)
    assert isinstance(m.fs.battery.nameplate_energy, Var)
    assert isinstance(m.fs.battery.elec_in, Var)
    assert isinstance(m.fs.battery.elec_out, Var)

    # Add Battery Unit Model
    m.fs.battery.dt.set_value(1)
    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    m.fs.battery.nameplate_power.set_value(5)
    m.fs.battery.nameplate_energy.fix(20)
    m.fs.battery.elec_in.fix(5)
    m.fs.battery.elec_out.fix(0)

    assert_units_consistent(m)

    solver = SolverFactory('ipopt')
    solver.solve(m.fs)

    assert m.fs.battery.elec_in.value == 5
    assert m.fs.battery.elec_out.value == 0
    assert m.fs.battery.state_of_charge.value == 5.0
    assert m.fs.battery.energy_throughput.value == 2.5

