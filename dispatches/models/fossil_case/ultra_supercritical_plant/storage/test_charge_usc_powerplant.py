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

"""Test for ultra supercritical power plant integrated with charge
storage system

"""

__author__ = "Soraya Rawlings"

import pytest

from pyomo.environ import TerminationCondition, value, SolverFactory
from pyomo.util.check_units import assert_units_consistent
from pyomo.contrib.fbbt.fbbt import  _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression

from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver

from dispatches.models.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)
from . import charge_design_ultra_supercritical_power_plant as charge_usc


optarg = {"max_iter": 300,
          "tol": 1e-8,
          "halt_on_ampl_error": "yes"}
solver = get_solver('ipopt', optarg)
add_efficiency = True
power_max = 436
heat_duty = 150


@pytest.fixture(scope="module")
def model():

    # Build ultra-supercritical plant base model
    m_usc = usc.build_plant_model()

    # Initialize ultra-supercritical plant base model
    usc.initialize(m_usc)

    m = charge_usc.create_charge_model(m_usc,
                                       add_efficiency=add_efficiency,
                                       power_max=power_max)

    # Give all the required inputs to the model
    charge_usc.set_model_input(m)

    # Add scaling factor
    charge_usc.set_scaling_factors(m)

    return m


@pytest.mark.integration
def test_main_function():

    # Build ultra-supercritical plant base model
    m_usc = usc.build_plant_model()

    # Initialize ultra-supercritical plant base model
    usc.initialize(m_usc)

    # Build charge model
    m = charge_usc.main(m_usc, solver=solver, optarg=optarg)

    # Solve design optimization problem
    charge_usc.model_analysis(m, heat_duty=heat_duty)

    # Solve model using GDPopt
    results = charge_usc.run_gdp(m)

    # Print results
    charge_usc.print_results(m, results)


@pytest.mark.integration
def test_initialize(model):
    # Check that the charge model is initialized properly and has 0
    # degrees of freedom
    charge_usc.initialize(model, solver=solver, optarg=optarg)
    assert degrees_of_freedom(model) == 0


@pytest.mark.integration
def test_costing(model):
    # Check that the cost correlations in charge model are initialized
    # properly and have 0 degrees of freedom
    charge_usc.build_costing(model, solver=solver)
    assert degrees_of_freedom(model) == 0


@pytest.mark.integration
def test_usc_charge_model(model):
    # Add missing functions to complete the charge model (add bounds
    # and disjunctions)
    charge_usc.add_bounds(model, power_max=power_max)
    charge_usc.add_disjunction(model)

    # Add design optimization problem
    charge_usc.model_analysis(model, heat_duty=heat_duty)

    opt = SolverFactory('gdpopt')
    opt.CONFIG.strategy = 'RIC'
    opt.CONFIG.tee = False
    opt.CONFIG.mip_solver = 'cbc'
    opt.CONFIG.nlp_solver = 'ipopt'
    opt.CONFIG.init_strategy = "no_init"
    opt.CONFIG.subproblem_presolve = False
    _prop_bnds_root_to_leaf_map[
        ExternalFunctionExpression] = lambda x, y, z: None

    result = opt.solve(model)

    assert result.solver.termination_condition == TerminationCondition.optimal
    assert value(model.fs.charge.hp_source_disjunct.binary_indicator_var) == 1
    assert value(model.fs.charge.solar_salt_disjunct.binary_indicator_var) == 1
    assert value(
        model.fs.charge.solar_salt_disjunct.hxc.area) == pytest.approx(
            1838.2,
            abs=1e-1)


# TODO: Add the unit consistency check once the PR #2395 is merged in Pyomo
# @pytest.mark.integration
# def test_unit_consistency(model):
#     assert_units_consistent(model)
