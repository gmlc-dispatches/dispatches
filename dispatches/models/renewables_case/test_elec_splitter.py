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
# Import objects from pyomo package
from pytest import approx
from pyomo.environ import ConcreteModel, SolverFactory, Var, TerminationCondition, SolverStatus, value

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock
from idaes.core.util.testing import initialization_tester

from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter


def test_elec_splitter_num_outlets_build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})  # dynamic or ss flowsheet needs to be specified here

    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3})

    assert hasattr(m.fs.unit, "electricity")
    assert hasattr(m.fs.unit, "electricity_in")
    assert hasattr(m.fs.unit, "outlet_list")
    assert isinstance(m.fs.unit.electricity, Var)
    assert isinstance(m.fs.unit.outlet_1_elec, Var)
    assert isinstance(m.fs.unit.outlet_2_elec, Var)
    assert isinstance(m.fs.unit.outlet_3_elec, Var)


def test_elec_splitter_num_outlets_init_0():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3})

    # fix 2 outlets, dof=0
    m.fs.unit.electricity_in.electricity.fix(1)
    m.fs.unit.outlet_1_elec.fix(0.25)
    m.fs.unit.outlet_2_elec.fix(0.25)
    initialization_tester(m)


def test_elec_splitter_num_outlets_init_1():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3})

    # fix 1 outlets, dof=1
    m.fs.unit.electricity_in.electricity.fix(1)
    m.fs.unit.outlet_1_elec.fix(0.25)
    initialization_tester(m, dof=1)


def test_elec_splitter_num_outlets_init_3():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3,
                                            "add_split_fraction_vars": True})

    # fix 2 outlets, dof=0
    m.fs.unit.electricity_in.electricity.fix(1)
    m.fs.unit.split_fraction['outlet_1', 0].fix(0.25)
    m.fs.unit.split_fraction['outlet_2', 0].fix(0.25)
    initialization_tester(m)


def test_elec_splitter_num_outlets_init_4():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3,
                                            "add_split_fraction_vars": True})

    # fix 1 outlets, dof=1
    m.fs.unit.electricity_in.electricity.fix(1)
    m.fs.unit.split_fraction['outlet_1', 0].fix(0.25)
    initialization_tester(m, dof=1)


def test_elec_splitter_num_outlets_solve_0():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3})

    m.fs.unit.electricity_in.electricity.fix(1)
    m.fs.unit.outlet_1_elec.fix(0.25)
    m.fs.unit.outlet_2_elec.fix(0.25)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.unit.outlet_1_elec[0].value == 0.25
    assert m.fs.unit.outlet_2_elec[0].value == 0.25
    assert m.fs.unit.outlet_3_elec[0].value == approx(0.5, 1e-4)


def test_elec_splitter_num_outlets_solve_1():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.unit = ElectricalSplitter(default={"num_outlets": 3})

    m.fs.unit.outlet_1_elec.fix(25)
    m.fs.unit.outlet_2_elec.fix(25)
    m.fs.unit.outlet_3_elec.fix(50)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.unit.electricity[0].value == 100


def test_elec_splitter_outlet_list():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})  # dynamic or ss flowsheet needs to be specified here

    m.fs.unit = ElectricalSplitter(default={"outlet_list": ["o1", "o2"]})

    assert hasattr(m.fs.unit, "electricity")
    assert hasattr(m.fs.unit, "electricity_in")
    assert hasattr(m.fs.unit, "split_fraction")
    assert hasattr(m.fs.unit, "outlet_list")
    assert isinstance(m.fs.unit.electricity, Var)
    assert isinstance(m.fs.unit.o1_elec, Var)
    assert isinstance(m.fs.unit.o2_elec, Var)

    m.fs.unit.o1_elec.fix(0.5)
    m.fs.unit.o2_elec.fix(0.5)

    solver = SolverFactory('ipopt')
    results = solver.solve(m.fs)

    assert results.solver.termination_condition == TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert m.fs.unit.electricity[0].value == approx(1, 1e-4)
    assert value(m.fs.unit.split_fraction['o1', 0]) == approx(0.5, 1e-4)
    assert value(m.fs.unit.split_fraction['o2', 0]) == approx(0.5, 1e-4)

    m.fs.unit.report(dof=True)

