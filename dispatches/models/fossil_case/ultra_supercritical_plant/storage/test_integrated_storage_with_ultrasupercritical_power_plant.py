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

"""Test for integrated storage with ultrasupercritical power plant model

"""

__author__ = "Naresh Susarla"

import pytest

from pyomo.environ import value
from pyomo.util.check_units import assert_units_consistent

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver

from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
from . import integrated_storage_with_ultrasupercritical_power_plant as isp

@pytest.fixture(scope="module")
def model():

    # Build ultra-supercritical plant base model
    m = usc.build_plant_model()
    usc.initialize(m)
    # Create a flowsheet, add properties, unit models, and arcs
    m = isp.create_integrated_model(m, max_power=436)

    # Give all the required inputs to the model
    isp.set_model_input(m)

    # Add scaling factor
    isp.set_scaling_factors(m)

    return m


@pytest.mark.integration
def test_build(model):

    if degrees_of_freedom(model) != 0:
        raise Exception("Degrees of freedom after model build is not zero")

@pytest.mark.integration
def test_initialization(model):
    # Check that the cost correlations in charge model are initialized
    # properly and have 0 degrees of freedom
    isp.initialize(model)

    if degrees_of_freedom(model) != 0:
        raise Exception("Degrees of freedom after model initialize is not zero")


@pytest.mark.integration
def test_costing_method(model):
    # Add missing functions to complete the charge model (add bounds
    # and disjunctions)
    m = isp.build_costing(model)

    # Initialize with bounds
    isp.initialize_with_costing(m)
    isp.add_bounds(m)

    if degrees_of_freedom(m) != 0:
        raise Exception("Degrees of freedom after model initialize is not zero")

@pytest.mark.integration
def test_main_function():

    optarg = {"max_iter": 300}
    solver = get_solver('ipopt', optarg)


    model = isp.main(max_power=436)

    model = isp.model_analysis(
        model,
        solver,
        power=460,
        max_power=436,
        tank_scenario="hot_empty",
        fix_power=False)

    assert value(model.fs.revenue) == pytest.approx(9627.76, abs=1e-1)
    assert value(model.obj) == pytest.approx(5.17, abs=1e-1)
    assert value(model.fs.hxd.area) == pytest.approx(2204.88,
                                                     abs=1e-1)

# TODO: Add the unit consistency check once the PR #2395 is merged in Pyomo
# @pytest.mark.integration
# def test_unit_consistency(m):
#     assert_units_consistent(m)
