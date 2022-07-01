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
"""
Basic tests for H2 property package
"""
import pytest

from pyomo.environ import ConcreteModel, value, SolverFactory, \
    TerminationCondition, SolverStatus

from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration

from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property \
    import GenericParameterBlock
from idaes.core.solvers import get_solver


def test_h2_props():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    m.fs.props = GenericParameterBlock(default=configuration)

    m.fs.state = m.fs.props.build_state_block(
        m.fs.config.time, default={"defined_state": True})

    # Fix state
    m.fs.state[0].flow_mol.fix(1)
    m.fs.state[0].mole_frac_comp.fix(1)
    m.fs.state[0].temperature.fix(300)
    m.fs.state[0].pressure.fix(101325)

    # Initialize state
    m.fs.state.initialize()

    solver = get_solver()
    results = solver.solve(m.fs)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    # Verify against NIST tables
    assert value(m.fs.state[0].cp_mol) == pytest.approx(28.85, rel=1e-2)
    assert value(m.fs.state[0].enth_mol) == pytest.approx(53.51, rel=1e-2)
    assert value(m.fs.state[0].entr_mol) == pytest.approx(130.9, rel=1e-2)
    assert (value(m.fs.state[0].gibbs_mol/m.fs.state[0].temperature) ==
            pytest.approx(-130.7, rel=1e-2))

    # Try another temeprature
    m.fs.state[0].temperature.fix(500)

    results = solver.solve(m.fs)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert value(m.fs.state[0].cp_mol) == pytest.approx(29.26, rel=1e-2)
    assert value(m.fs.state[0].enth_mol) == pytest.approx(5880, rel=1e-2)
    assert value(m.fs.state[0].entr_mol) == pytest.approx(145.7, rel=1e-2)
    assert (value(m.fs.state[0].gibbs_mol/m.fs.state[0].temperature) ==
            pytest.approx(-134.0, rel=1e-2))

    # Try another temeprature
    m.fs.state[0].temperature.fix(900)

    results = solver.solve(m.fs)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert value(m.fs.state[0].cp_mol) == pytest.approx(29.88, rel=1e-2)
    assert value(m.fs.state[0].enth_mol) == pytest.approx(17680, rel=1e-2)
    assert value(m.fs.state[0].entr_mol) == pytest.approx(163.1, rel=1e-2)
    assert (value(m.fs.state[0].gibbs_mol/m.fs.state[0].temperature) ==
            pytest.approx(-143.4, rel=1e-2))
