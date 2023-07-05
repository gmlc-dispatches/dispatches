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
# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, Var, Objective
from pyomo.util.check_units import assert_units_consistent

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock
from dispatches.unit_models import SolarPV


def test_pv():
    # Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)   # dynamic or ss flowsheet needs to be specified here

    pv_capacity_factors = {'capacity_factor': [0.5]}

    m.fs.unit = SolarPV(**pv_capacity_factors)
    assert hasattr(m.fs.unit, "capacity_factor")
    assert hasattr(m.fs.unit, "electricity_out")
    assert isinstance(m.fs.unit.system_capacity, Var)
    assert isinstance(m.fs.unit.electricity, Var)

    m.fs.unit.system_capacity.fix(50000) # kW

    assert_units_consistent(m)

    m.fs.unit.initialize()

    assert m.fs.unit.capacity_factor[0].value == pytest.approx(0.5, rel=1e-2)
    assert m.fs.unit.electricity_out.electricity[0].value == pytest.approx(25000, rel=1e-2)

    solver = SolverFactory('ipopt')
    solver.solve(m.fs)

    assert m.fs.unit.capacity_factor[0].value == pytest.approx(0.5, rel=1e-2)
    assert m.fs.unit.electricity_out.electricity[0].value <= 25000
