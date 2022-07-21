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
Nuclear Flowsheet Tester
Author: Konor Frick
Date: May 11, 2021
"""

import pytest
from pyomo.environ import (ConcreteModel,
                           TerminationCondition,
                           SolverStatus)
from pyomo.util.check_units import assert_units_consistent

from dispatches.models.nuclear_case.flowsheets.nuclear_flowsheet \
    import build_ne_flowsheet, fix_dof_and_initialize
from idaes.core.solvers import get_solver


solver = get_solver()


def test_nuclear_fs():
    m = ConcreteModel()

    # Build the nuclear flowsheet
    build_ne_flowsheet(m)

    # Fix the degrees of freedom and initialize
    fix_dof_and_initialize(m)

    results = solver.solve(m)

    # Ensure that units are consistent
    assert_units_consistent(m)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    # PEM System
    assert m.fs.pem.outlet.flow_mol[0].value == \
        pytest.approx(505.481, rel=1e-1)
    assert m.fs.pem.outlet.temperature[0].value == \
        pytest.approx(300, rel=1e-1)
    assert m.fs.pem.outlet.pressure[0].value == \
        pytest.approx(101325, rel=1e-1)
    
    # Hydrogen tank
    assert m.fs.h2_tank.tank_holdup_previous[0].value == \
        pytest.approx(0, rel = 1e-1)
    assert m.fs.h2_tank.tank_holdup[0].value == \
        pytest.approx(1747732.3199, rel=1e-1)

    # Hydrogen Turbine
    # Compressor
    assert m.fs.h2_turbine.compressor.outlet.temperature[0].value == \
        pytest.approx(793.42, rel=1e-1)

    # Stoichiometric Reactor
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'hydrogen'].value == \
        pytest.approx(0.00088043, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'nitrogen'].value == \
        pytest.approx(0.73278, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'oxygen'].value == \
        pytest.approx(0.15276, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'water'].value == \
        pytest.approx(0.1103, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'argon'].value == \
        pytest.approx(0.0032773, rel=1e-1)

    # Turbine
    assert m.fs.h2_turbine.turbine.inlet.temperature[0].value == \
        pytest.approx(1451.5, rel=1e-1)
    assert m.fs.h2_turbine.turbine.outlet.temperature[0].value == \
        pytest.approx(739.3, rel=1e-1)

