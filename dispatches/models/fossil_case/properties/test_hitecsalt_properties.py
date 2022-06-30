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
Tests for Hitec salt property package.
Authors: Naresh Susarla
"""

import pytest
from pyomo.environ import (ConcreteModel,
                           Constraint,
                           Expression,
                           Param,
                           value,
                           Var)
from pyomo.util.check_units import assert_units_consistent
from idaes.core import (MaterialBalanceType,
                        EnergyBalanceType)
import idaes.logger as idaeslog
from . import hitecsalt_properties


@pytest.fixture(scope="class")
def model():
    model = ConcreteModel()
    model.params = hitecsalt_properties.HitecsaltParameterBlock()
    model.props = model.params.build_state_block([1])
    return model


@pytest.mark.unit
def test_config(model):
    assert len(model.params.config) == 1


@pytest.mark.unit
def test_build(model):

    assert len(model.params.phase_list) == 1
    for i in model.params.phase_list:
        assert i == "Liq"

    assert len(model.params.component_list) == 1
    for i in model.params.component_list:
        assert i in ['Hitec_Salt']

    assert isinstance(model.params.cp_param_1, Param)
    assert value(model.params.cp_param_1) == 5806

    assert isinstance(model.params.cp_param_2, Param)
    assert value(model.params.cp_param_2) == -10.833

    assert isinstance(model.params.cp_param_3, Param)
    assert value(model.params.cp_param_3) == 7.2413E-3

    assert isinstance(model.params.rho_param_1, Param)
    assert value(model.params.rho_param_1) == 2293.6

    assert isinstance(model.params.rho_param_2, Param)
    assert value(model.params.rho_param_2) == -0.7497

    assert isinstance(model.params.mu_param_1, Param)
    assert value(model.params.mu_param_1) == -4.343

    assert isinstance(model.params.mu_param_2, Param)
    assert value(model.params.mu_param_2) == -2.0143

    assert isinstance(model.params.mu_param_3, Param)
    assert value(model.params.mu_param_3) == -5.011

    assert isinstance(model.params.kappa_param_1, Param)
    assert value(model.params.kappa_param_1) == 0.421

    assert isinstance(model.params.kappa_param_2, Param)
    assert value(model.params.kappa_param_2) == -6.53E-4

    assert isinstance(model.params.kappa_param_3, Param)
    assert value(model.params.kappa_param_3) == -260

    assert isinstance(model.params.ref_temperature, Param)
    assert value(model.params.ref_temperature) == 298.15

    assert isinstance(model.props[1].flow_mass, Var)
    assert value(model.props[1].flow_mass) == 100

    assert isinstance(model.props[1].pressure, Var)
    assert value(model.props[1].pressure) == 1.01325E5

    assert isinstance(model.props[1].temperature, Var)
    assert value(model.props[1].temperature) == 550

    assert isinstance(model.props[1].enth_mass, Var)
    assert len(model.props[1].enth_mass) == 1
    for i in model.props[1].enth_mass:
        assert value(model.props[1].enth_mass[i]) == 1

    assert isinstance(model.props[1].enthalpy_eq, Constraint)
    assert isinstance(model.props[1].cp_mass, Expression)
    assert isinstance(model.props[1].dens_mass, Expression)
    assert isinstance(model.props[1].visc_d_phase, Expression)
    assert isinstance(model.props[1].therm_cond_phase, Expression)


@pytest.mark.unit
def test_get_material_flow_terms(model):
    for p in model.params.phase_list:
        for j in model.params.component_list:
            assert str(
                model.props[1].get_material_flow_terms(p, j)) == str(
                model.props[1].flow_mass)


@pytest.mark.unit
def test_get_enthalpy_flow_terms(model):
    for p in model.params.phase_list:
        assert str(model.props[1].get_enthalpy_flow_terms(p)) == str(
                model.props[1].enthalpy_flow_terms[p])


@pytest.mark.unit
def test_default_material_balance_type(model):
    assert model.props[1].default_material_balance_type() == \
        MaterialBalanceType.componentTotal


@pytest.mark.unit
def test_default_energy_balance_type(model):
    assert model.props[1].default_energy_balance_type() == \
        EnergyBalanceType.enthalpyTotal


@pytest.mark.unit
def test_define_state_vars(model):
    sv = model.props[1].define_state_vars()

    assert len(sv) == 3
    for i in sv:
        assert i in ["flow_mass",
                     "temperature",
                     "pressure"]


@pytest.mark.unit
def test_define_port_members(model):
    sv = model.props[1].define_state_vars()

    assert len(sv) == 3
    for i in sv:
        assert i in ["flow_mass",
                     "temperature",
                     "pressure"]


@pytest.mark.unit
def test_initialize(model):
    assert not model.props[1].flow_mass.fixed
    assert not model.props[1].temperature.fixed
    assert not model.props[1].pressure.fixed

    model.props.initialize(hold_state=False, outlvl=idaeslog.INFO)

    assert not model.props[1].flow_mass.fixed
    assert not model.props[1].temperature.fixed
    assert not model.props[1].pressure.fixed


@pytest.mark.unit
def test_initialize_hold(model):
    assert not model.props[1].flow_mass.fixed
    assert not model.props[1].temperature.fixed
    assert not model.props[1].pressure.fixed

    flags = model.props.initialize(hold_state=True)

    assert model.props[1].flow_mass.fixed
    assert model.props[1].temperature.fixed
    assert model.props[1].pressure.fixed

    model.props.release_state(flags, outlvl=idaeslog.INFO)

    assert not model.props[1].flow_mass.fixed
    assert not model.props[1].temperature.fixed
    assert not model.props[1].pressure.fixed


@pytest.mark.unit
def check_units(model):
    assert_units_consistent(model)
