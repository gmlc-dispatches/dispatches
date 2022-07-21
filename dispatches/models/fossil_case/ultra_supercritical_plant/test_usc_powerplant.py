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
Test for ultra supercritical power plant flowsheet
"""

__author__ = "Naresh Susarla"

from unittest.mock import patch, MagicMock
import runpy
import pytest
from pyomo.environ import TerminationCondition, value
from pyomo.util.check_units import assert_units_consistent

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver

from . import ultra_supercritical_powerplant as usc

solver = get_solver()


@pytest.fixture(scope="module")
def model():
    m = usc.build_plant_model()
    return m


@pytest.mark.integration
def test_initialize(model):
    # check that the model initialized properly and has 0 degrees of freedom
    usc.initialize(model)
    assert(degrees_of_freedom(model) == 0)


@pytest.mark.integration
def test_unit_consistency(model):
    assert_units_consistent(model)


@pytest.mark.integration
def test_main_function():
    # Build ultra supercriticla power plant model
    m = usc.build_plant_model()

    # Initialize the model (sequencial initialization and custom routines)
    usc.initialize(m)

    # Ensure after the model is initialized, the degrees of freedom = 0
    assert degrees_of_freedom(m) == 0

    # User can import the model from build_plant_model for analysis
    # A sample analysis function is called below
    m_result = usc.model_analysis(m, solver)

    # View results in a process flow diagram
    usc.view_result("pfd_usc_powerplant_result.svg", m_result)


@pytest.mark.integration
def test_usc_model(model):
    result = solver.solve(model, tee=False)
    assert result.solver.termination_condition == TerminationCondition.optimal
    assert (value(model.fs.plant_power_out[0]) ==
            pytest.approx(436.466,
                          abs=1e-2))  # Ref: Report/USDOE/FE-0400"
    assert (value(model.fs.constraint_bfp_power[0]) ==
            pytest.approx(0,
                          abs=1e-2))


@pytest.mark.integration
def test_change_power(model):
    model.fs.plant_power_out[0].fix(300)
    model.fs.boiler.inlet.flow_mol[0].unfix()
    result = solver.solve(model, tee=False)
    assert result.solver.termination_condition == TerminationCondition.optimal
    assert (value(model.fs.boiler.inlet.flow_mol[0]) ==
            pytest.approx(12474.4,
                          abs=1e-2))


@pytest.mark.integration
def test_change_pressure(model):
    model.fs.plant_power_out[0].unfix()
    model.fs.boiler.inlet.flow_mol[0].fix(17854)
    model.fs.boiler.outlet.pressure.fix(27e6)
    result = solver.solve(model, tee=False)
    assert result.solver.termination_condition == TerminationCondition.optimal
    assert (value(model.fs.plant_power_out[0]) ==
            pytest.approx(446.15,
                          abs=1e-2))
    assert (value(model.fs.plant_heat_duty[0]) ==
            pytest.approx(940.4,
                          abs=1e-2))
