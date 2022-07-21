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
# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var, Objective
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
        resource_timeseries[time] = ((10, 180, 1),)

    wind_config = {'resource_probability_density': resource_timeseries}

    m.fs.unit = Wind_Power(default=wind_config)
    assert hasattr(m.fs.unit, "capacity_factor")
    assert hasattr(m.fs.unit, "electricity_out")
    assert isinstance(m.fs.unit.system_capacity, Var)
    assert isinstance(m.fs.unit.electricity, Var)

    m.fs.unit.system_capacity.fix(50000) # kW

    assert_units_consistent(m)

    m.fs.unit.initialize()

    assert m.fs.unit.capacity_factor[0].value == pytest.approx(0.5755, rel=1e-2)
    assert m.fs.unit.electricity_out.electricity[0].value == pytest.approx(28775.06, rel=1e-2)

    solver = SolverFactory('ipopt')
    solver.solve(m.fs)

    assert m.fs.unit.capacity_factor[0].value == pytest.approx(0.5755, rel=1e-2)
    assert m.fs.unit.electricity_out.electricity[0].value <= 28775.06


def test_windpower2():
    # Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})   # dynamic or ss flowsheet needs to be specified here

    # ((wind m/s, wind degrees from north clockwise, probability), )
    resource_speed = [10 for _ in list(m.fs.config.time.data())]

    wind_config = {'resource_speed': resource_speed}

    m.fs.unit = Wind_Power(default=wind_config)

    m.fs.unit.system_capacity.fix(50000) # kW
    m.fs.unit.initialize()
    m.fs.obj = Objective(expr=-m.fs.unit.electricity_out.electricity[0])

    solver = SolverFactory('ipopt')
    solver.solve(m.fs)

    assert m.fs.unit.electricity_out.electricity[0].value == pytest.approx(30083.39, rel=1e-2)
