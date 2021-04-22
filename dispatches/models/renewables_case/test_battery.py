import pytest
# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock

from idaes.core.util.model_statistics import degrees_of_freedom, unfixed_variables_in_activated_equalities_set, activated_equalities_set

from dispatches.models.renewables_case.battery import BatteryStorage


def test_battery():
    # Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": True})   # dynamic or ss flowsheet needs to be specified here

    m.fs.battery = BatteryStorage()
    assert hasattr(m.fs.battery, "dt")

    # Add Battery Unit Model
    m.fs.battery.dt.set_value(1)
    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    m.fs.battery.nameplate_power.fix(5)
    m.fs.battery.nameplate_energy.fix(20)
    m.fs.battery.elec_in[0].fix(5)
    m.fs.battery.elec_in[1].fix(0)
    m.fs.battery.elec_out[0].fix(0)
    m.fs.battery.elec_out[1].fix(5)

    assert_units_consistent(m)

    solver = SolverFactory('ipopt')
    solver.solve(m.fs)

    assert m.fs.battery.elec_in[0].value == 5
    assert m.fs.battery.elec_out[0].value == 0
    assert m.fs.battery.state_of_charge[0].value == 5.0
    assert m.fs.battery.energy_throughput[0].value == 2.5

