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

import pytest
from pyomo.environ import (ConcreteModel, TerminationCondition,
                           SolverStatus, value, units as pyunits)

from idaes.core import FlowsheetBlock
from heat_exchanger_tube import ConcreteTubeSide as HX1D
from idaes.generic_models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern

from idaes.generic_models.properties import iapws95

from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.util.testing import get_default_solver


def test_build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})

    m.fs.properties = iapws95.Iapws95ParameterBlock(default={
        "phase_presentation": iapws95.PhaseType.LG})

    m.fs.unit = HX1D(default={
            "tube_side": {"property_package": m.fs.properties},
            "flow_type": HeatExchangerFlowPattern.cocurrent})

    m.fs.unit.d_tube_outer.fix(0.01167)
    m.fs.unit.d_tube_inner.fix(0.01167)
    m.fs.unit.tube_length.fix(4.85)
    m.fs.unit.tube_heat_transfer_coefficient.fix(500)

    m.fs.unit.tube_inlet.flow_mol[0].fix(1)  # mol/s
    m.fs.unit.tube_inlet.pressure[0].fix(101325)  # Pa
    m.fs.unit.tube_inlet.enth_mol[0].\
        fix(iapws95.htpx(300*pyunits.K, 101325*pyunits.Pa))  # K

    m.fs.unit.temperature_wall[0, :].fix(1000)

    assert degrees_of_freedom(m) == 0

    m.fs.unit.initialize()

    solver = get_default_solver()
    results = solver.solve(m)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    assert pytest.approx(1, abs=1e-5) == \
        value(m.fs.unit.tube_outlet.flow_mol[0])

    assert pytest.approx(55702.16, abs=1e0) == \
        value(m.fs.unit.tube_outlet.enth_mol[0])

    assert pytest.approx(101325, abs=1e2) == \
        value(m.fs.unit.tube_outlet.pressure[0])

    assert pytest.approx(581.80, abs=1e2) == \
        value(m.fs.unit.tube.properties[0, 1].temperature)

