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
Turbo-Generator Set for a Hydrogen turbine.

Compressor -> Stoichiometric Reactor -> Turbine
Author: Konor Frick
Date: April 2, 2021
Notes: it is noted that in this example the hydrogen is compressed along
with the air in the compressor as opposed to having a separate fuel
injection system. Noting this is a simplified version of the H2 turbine.
"""

import pytest
from pyomo.environ import ConcreteModel, SolverFactory, \
    value, TerminationCondition, SolverStatus
from idaes.core import FlowsheetBlock


from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration
import dispatches.models.nuclear_case.properties.h2_reaction \
    as reaction_props
from dispatches.models.nuclear_case.unit_models.\
    hydrogen_turbine_unit import HydrogenTurbine
from idaes.models.properties.modular_properties.base.generic_property \
    import GenericParameterBlock


from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver


def test_build():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Air Properties
    m.fs.properties1 = GenericParameterBlock(default=configuration)

    m.fs.reaction_params = reaction_props.\
        H2ReactionParameterBlock(default={"property_package":
                                          m.fs.properties1})

    # Adding H2 turbine model
    m.fs.h2_turbine = HydrogenTurbine(
                default={"property_package": m.fs.properties1,
                         "reaction_package": m.fs.reaction_params})

    # Check build
    assert hasattr(m.fs.h2_turbine, "compressor")
    assert hasattr(m.fs.h2_turbine, "stoic_reactor")
    assert hasattr(m.fs.h2_turbine, "turbine")

    assert degrees_of_freedom(m) == 13

    # Inlet Conditions********************************************************
    # ************************************************************************
    # Inlet Conditions of the inlet to the compressor.
    m.fs.h2_turbine.compressor.inlet.flow_mol[0].fix(4135.2)
    m.fs.h2_turbine.compressor.inlet.temperature[0].fix(288.15)
    m.fs.h2_turbine.compressor.inlet.pressure[0].fix(101325)

    m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "oxygen"].fix(0.188)
    m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "argon"].fix(0.003)
    m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "nitrogen"].fix(0.702)
    m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "water"].fix(0.022)
    m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "hydrogen"].fix(0.085)

    m.fs.h2_turbine.compressor.deltaP.fix(2.401e6)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

    # Specify the Stoichiometric Conversion Rate of hydrogen
    # in the equation shown below
    # H2(g) + 1/2 O2(g) --> H2O(g) + energy
    # Complete Combustion
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    # Turbine Parameters
    m.fs.h2_turbine.turbine.deltaP.fix(-2.401e6)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    assert degrees_of_freedom(m) == 0

    solver = get_solver()

    # Begin Initialization and solve for the system.
    m.fs.h2_turbine.initialize()

    results = solver.solve(m, tee=False)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    # Compressor
    assert value(m.fs.h2_turbine.compressor.outlet.temperature[0]) == \
        pytest.approx(763.25, rel=1e-1)

    # Stoichiometric Reactor
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'hydrogen']) == \
        pytest.approx(0.00085, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'nitrogen']) == \
        pytest.approx(0.73285, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'oxygen']) == \
        pytest.approx(0.15232, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'water']) == \
        pytest.approx(0.11085, rel=1e-1)
    assert value(m.fs.h2_turbine.stoic_reactor.
                 outlet.mole_frac_comp[0, 'argon']) == \
        pytest.approx(0.0031318, rel=1e-1)

    # Turbine
    assert value(m.fs.h2_turbine.turbine.inlet.temperature[0]) == \
        pytest.approx(1426.3, rel=1e-1)
    assert value(m.fs.h2_turbine.turbine.outlet.temperature[0]) == \
        pytest.approx(726.44, rel=1e-1)

    m.fs.h2_turbine.report()
