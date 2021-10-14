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
"""
Nuclear Flowsheet Tester
Author: Konor Frick
Date: May 11, 2021
"""

import pytest
from pyomo.environ import value, TerminationCondition, SolverStatus

from dispatches.models.nuclear_case.flowsheets.Nuclear_flowsheet \
    import create_model, set_inputs, initialize_model
from idaes.core.util import get_solver


solver = get_solver()


def test_nuclear_fs():
    m = create_model()
    m = set_inputs(m)
    m = initialize_model(m)

    results = solver.solve(m)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    # PEM System
    assert m.fs.pem.outlet.flow_mol[0].value == \
        pytest.approx(252.740, rel=1e-1)
    assert m.fs.pem.outlet.temperature[0].value == \
        pytest.approx(300, rel=1e-1)
    assert m.fs.pem.outlet.pressure[0].value == \
        pytest.approx(101325, rel=1e-1)

    # Hydrogen Turbine
    # Compressor
    assert value(m.fs.h2_turbine.compressor.outlet.temperature[0]) == \
        pytest.approx(765.8, rel=1e-1)

    # Stoichiometric Reactor
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'hydrogen']) == \
        pytest.approx(0.00086382, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'nitrogen']) == \
        pytest.approx(0.73193, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'oxygen']) == \
        pytest.approx(0.15143, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'water']) == \
        pytest.approx(0.11249, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'argon']) == \
        pytest.approx(0.0032793, rel=1e-1)

    # Turbine
    assert value(m.fs.h2_turbine.turbine.inlet.temperature[0]) == \
        pytest.approx(1440, rel=1e-1)
    assert value(m.fs.h2_turbine.turbine.outlet.temperature[0]) == \
        pytest.approx(733.76, rel=1e-1)

