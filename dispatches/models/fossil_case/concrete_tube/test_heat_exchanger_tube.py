##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Tests for ConcreteTubeSide model.

Author: Konica Mulani, Jaffer Ghouse
"""

import pprint
import sys
pprint.pprint(sys.path)
import pytest
from pyomo.environ import (ConcreteModel, TerminationCondition,
                           SolverStatus, value, units as pyunits)
from pyomo.common.config import ConfigBlock
from pyomo.util.check_units import (assert_units_consistent,
                                    assert_units_equivalent)

from idaes.core import (FlowsheetBlock, MaterialBalanceType, EnergyBalanceType,
                        MomentumBalanceType, useDefault)
from heat_exchanger_tube import ConcreteTubeSide as HX1D
from idaes.generic_models.unit_models.heat_exchanger import HeatExchangerFlowPattern

from idaes.generic_models.properties.core.generic.generic_property import (
        GenericParameterBlock)
from idaes.generic_models.properties.core.examples.BT_PR import \
    configuration
from idaes.generic_models.properties.activity_coeff_models.BTX_activity_coeff_VLE \
    import BTXParameterBlock
from idaes.generic_models.properties import iapws95
from idaes.generic_models.properties.examples.saponification_thermo import (
    SaponificationParameterBlock)

from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              number_variables,
                                              number_total_constraints,
                                              number_unused_variables)
from idaes.core.util.testing import (get_default_solver,
                                     PhysicalParameterTestBlock,
                                     initialization_tester)
from idaes.core.util import scaling as iscale


# Imports to assemble BT-PR with different units
from idaes.core import LiquidPhase, VaporPhase, Component
from idaes.generic_models.properties.core.state_definitions import FTPx
from idaes.generic_models.properties.core.eos.ceos import Cubic, CubicType
from idaes.generic_models.properties.core.phase_equil import smooth_VLE
from idaes.generic_models.properties.core.phase_equil.bubble_dew import \
        LogBubbleDew
from idaes.generic_models.properties.core.phase_equil.forms import log_fugacity
import idaes.generic_models.properties.core.pure.RPP as RPP

import matplotlib.pyplot as plt




def test_fail():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})

    m.fs.properties = BTXParameterBlock(default={"valid_phase": 'Liq'})

    m.fs.unit = HX1D(default={
            "tube_side": {"property_package": m.fs.properties},
            "flow_type": HeatExchangerFlowPattern.cocurrent})

    m.fs.unit.d_tube_outer.fix(0.01167)
    m.fs.unit.d_tube_inner.fix(0.01167)
    m.fs.unit.tube_length.fix(4.85)
    m.fs.unit.tube_heat_transfer_coefficient.fix(51000)
    m.fs.unit.tube_inlet.flow_mol[0].fix(1)  # mol/s
    m.fs.unit.tube_inlet.temperature[0].fix(300)  # K
    m.fs.unit.tube_inlet.pressure[0].fix(101325)  # Pa
    m.fs.unit.tube_inlet.mole_frac_comp[0, "benzene"].fix(0.5)
    m.fs.unit.tube_inlet.mole_frac_comp[0, "toluene"].fix(0.5)
    m.fs.unit.temperature_wall[0,:].fix(500)

    # print("degrees_of_freedom b4:" , degrees_of_freedom(m))
    m.fs.unit.initialize()
    # print("degrees_of_freedom aft:" , degrees_of_freedom(m))
    solver = get_default_solver()
    results = solver.solve(m)
    # print(results)

    temp_profile = list(value(m.fs.unit.tube.properties[0.0,:].temperature))
    # print('temp_profile',temp_profile)
    len_tube = value(m.fs.unit.tube_length)
    length_domain = list(m.fs.unit.tube.length_domain)
    length_domain2 = [float(i)*len_tube for i in length_domain]
    # print('length_domain',length_domain2)
    # Generate plot
    # fig_1 = plt.figure(1)
    # plt.plot(length_domain2,temp_profile)

    assert value(m.fs.unit.d_tube_outer) == 0.01167
    assert value(m.fs.unit.d_tube_inner) == 0.01167
    assert value(m.fs.unit.tube_length) == 4.85
    #assert value(m.fs.unit.tube_heat_transfer_coefficient) == 51000
    assert value(m.fs.unit.tube_inlet.flow_mol[0]) == 1  # mol/s
    assert value(m.fs.unit.tube_inlet.temperature[0]) == 300  # K
    assert value(m.fs.unit.tube_inlet.pressure[0]) == 101325  # Pa
    assert value(m.fs.unit.tube_inlet.mole_frac_comp[0, "benzene"]) == 0.5
    assert value(m.fs.unit.tube_inlet.mole_frac_comp[0, "toluene"]) == 0.5
    #assert value(m.fs.unit.temperature_wall[0,:]) == 500