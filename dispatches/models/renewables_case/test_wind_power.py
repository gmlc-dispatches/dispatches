import pytest
# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var
from pyomo.util.check_units import assert_units_consistent

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock

from dispatches.models.renewables_case.wind_power import Wind_Power


def test_windpower():
    # Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})   # dynamic or ss flowsheet needs to be specified here

    # ((wind m/s, wind degrees from north clockwise, probability), )
    resource_timeseries = dict()
    for time in list(m.fs.config.time.data()):
        resource_timeseries[time] = ((10, 180, 0.5),
                                     (24, 180, 0.5))

    wind_config = {'resource_probability_density': resource_timeseries}

    m.fs.unit = Wind_Power(default=wind_config)

    assert hasattr(m.fs.unit, "capacity_factor")
    assert hasattr(m.fs.unit, "electricity_out")
    assert isinstance(m.fs.unit.system_capacity, Var)
    assert isinstance(m.fs.unit.electricity, Var)

    m.fs.unit.system_capacity.fix(50000) # kW

    assert_units_consistent(m)

    solver = SolverFactory('ipopt')
    solver.solve(m.fs)

    assert m.fs.unit.capacity_factor[0].value == pytest.approx(0.0001905, rel=1e-2)
    assert m.fs.unit.electricity_out.electricity[0].value == pytest.approx(9.525, rel=1e-2)

