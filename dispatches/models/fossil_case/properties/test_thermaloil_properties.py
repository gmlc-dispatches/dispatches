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

###############################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
###############################################################################
"""
Tests for Thermal Oil property package.
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
from idaes.core.util import get_solver
import idaes.logger as idaeslog
import thermaloil_properties


# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


class TestParamBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.params = thermaloil_properties.ThermalOilParameterBlock()

        return model

    @pytest.mark.unit
    def test_config(self, model):
        assert len(model.params.config) == 1

    @pytest.mark.unit
    def test_build(self, model):

        assert len(model.params.phase_list) == 1
        for i in model.params.phase_list:
            assert i == "Liq"

        assert len(model.params.component_list) == 1
        for i in model.params.component_list:
            assert i in ['therminol66']

        assert isinstance(model.params.cp_param_1, Param)
        assert value(model.params.cp_param_1) == 1496.005

        assert isinstance(model.params.cp_param_2, Param)
        assert value(model.params.cp_param_2) == 3.313

        assert isinstance(model.params.cp_param_3, Param)
        assert value(model.params.cp_param_3) == 0.0008970785

        assert isinstance(model.params.rho_param_1, Param)
        assert value(model.params.rho_param_1) == 1026.7

        assert isinstance(model.params.rho_param_2, Param)
        assert value(model.params.rho_param_2) == -0.7281

        assert isinstance(model.params.nu_param_1, Param)
        assert value(model.params.nu_param_1) == 586.375

        assert isinstance(model.params.nu_param_2, Param)
        assert value(model.params.nu_param_2) == 62.5

        assert isinstance(model.params.nu_param_3, Param)
        assert value(model.params.nu_param_3) == -2.2809

        assert isinstance(model.params.nu_param_4, Param)
        assert value(model.params.nu_param_4) == 1E-6

        assert isinstance(model.params.kappa_param_1, Param)
        assert value(model.params.kappa_param_1) == 0.118294

        assert isinstance(model.params.kappa_param_2, Param)
        assert value(model.params.kappa_param_2) == -3.3E-5

        assert isinstance(model.params.kappa_param_3, Param)
        assert value(model.params.kappa_param_3) == -1.5E-7

        assert isinstance(model.params.ref_temperature, Param)
        assert value(model.params.ref_temperature) == 273.15


class TestStateBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.params = thermaloil_properties.ThermalOilParameterBlock()

        model.props = model.params.build_state_block([1])

        return model

    @pytest.mark.unit
    def test_build(self, model):
        assert isinstance(model.props[1].flow_mass, Var)
        assert value(model.props[1].flow_mass) == 100

        assert isinstance(model.props[1].pressure, Var)
        assert value(model.props[1].pressure) == 1.01325E5

        assert isinstance(model.props[1].temperature, Var)
        assert value(model.props[1].temperature) == 523

        assert isinstance(model.props[1].enthalpy_mass, Var)
        assert len(model.props[1].enthalpy_mass) == 1
        for i in model.props[1].enthalpy_mass:
            assert value(model.props[1].enthalpy_mass[i]) == 1

        assert isinstance(model.props[1].enthalpy_eq, Constraint)
        assert isinstance(model.props[1].cp_mass, Expression)
        assert isinstance(model.props[1].density, Expression)
        assert isinstance(model.props[1].visc_kin, Expression)
        assert isinstance(model.props[1].therm_cond, Expression)

    @pytest.mark.unit
    def test_get_material_flow_terms(self, model):
        for p in model.params.phase_list:
            for j in model.params.component_list:
                assert str(
                    model.props[1].get_material_flow_terms(p, j)) == str(
                    model.props[1].flow_mass)

    @pytest.mark.unit
    def test_get_enthalpy_flow_terms(self, model):
        for p in model.params.phase_list:
            assert str(model.props[1].get_enthalpy_flow_terms(p)) == str(
                    model.props[1].enthalpy_flow_terms[p])

    @pytest.mark.unit
    def test_default_material_balance_type(self, model):
        assert model.props[1].default_material_balance_type() == \
            MaterialBalanceType.componentTotal

    @pytest.mark.unit
    def test_default_energy_balance_type(self, model):
        assert model.props[1].default_energy_balance_type() == \
            EnergyBalanceType.enthalpyTotal

    @pytest.mark.unit
    def test_define_state_vars(self, model):
        sv = model.props[1].define_state_vars()

        assert len(sv) == 3
        for i in sv:
            assert i in ["flow_mass",
                         "temperature",
                         "pressure"]

    @pytest.mark.unit
    def test_define_port_members(self, model):
        sv = model.props[1].define_state_vars()

        assert len(sv) == 3
        for i in sv:
            assert i in ["flow_mass",
                         "temperature",
                         "pressure"]

    @pytest.mark.unit
    def test_initialize(self, model):
        assert not model.props[1].flow_mass.fixed
        assert not model.props[1].temperature.fixed
        assert not model.props[1].pressure.fixed

        model.props.initialize(hold_state=False, outlvl=idaeslog.INFO)

        assert not model.props[1].flow_mass.fixed
        assert not model.props[1].temperature.fixed
        assert not model.props[1].pressure.fixed

    @pytest.mark.unit
    def test_initialize_hold(self, model):
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
    def check_units(self, model):
        assert_units_consistent(model)
