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
Basic tests for Therminol-66 property package
Author: Konor Frick and Jaffer Ghouse.
Date: February 16, 2021
Reference: “Therminol 66: High Performance Highly Stable Heat Transfer Fluid,
” Applied Chemistry, Creative Solutions, Solutia, Inc., St. Louis, Missouri;
https://www.therminol.com.
"""
import pytest
from pyomo.environ import ConcreteModel, value, SolverFactory, \
    TerminationCondition, SolverStatus
from idaes.core import FlowsheetBlock
from dispatches.models.fossil_case.thermal_oil.thermal_oil \
    import ThermalOilParameterBlock
from idaes.core.solvers import get_solver


def test_oil():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.therminol66_prop = ThermalOilParameterBlock()

    m.fs.state = m.fs.therminol66_prop.build_state_block(
        m.fs.config.time, default={"defined_state": True})

    # Fix state
    m.fs.state[0].flow_mass.fix(1)
    m.fs.state[0].temperature.fix(273.15+20)
    m.fs.state[0].pressure.fix(101325)

    # Initialize state
    m.fs.state.initialize()

    # Verify against Therminol Solutia tables
    assert value(m.fs.state[0].cp_mass) == pytest.approx(1562, rel=1e-1)
    assert value(m.fs.state[0].therm_cond) == pytest.approx(0.117574, rel=1e-1)
    assert value(m.fs.state[0].visc_kin) == pytest.approx(122.45, rel=1e-1)
    assert value(m.fs.state[0].density) == pytest.approx(1008.4, rel=1e-1)

    # Try another temperature
    m.fs.state[0].temperature.fix(273.15+180)

    solver = get_solver()
    results = solver.solve(m.fs)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert value(m.fs.state[0].cp_mass) == pytest.approx(2122, rel=1e-1)
    assert value(m.fs.state[0].therm_cond) == pytest.approx(0.107494, rel=1e-1)
    assert value(m.fs.state[0].visc_kin) == pytest.approx(1.17, rel=1e-1)
    assert value(m.fs.state[0].density) == pytest.approx(899.5, rel=1e-1)

    # Try another temperature
    m.fs.state[0].temperature.fix(273.15+350)

    results = solver.solve(m.fs)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert value(m.fs.state[0].cp_mass) == pytest.approx(2766, rel=1e-1)
    assert value(m.fs.state[0].therm_cond) == pytest.approx(0.088369, rel=1e-1)
    assert value(m.fs.state[0].visc_kin) == pytest.approx(0.42, rel=1e-1)
    assert value(m.fs.state[0].density) == pytest.approx(765.9, rel=1e-1)
