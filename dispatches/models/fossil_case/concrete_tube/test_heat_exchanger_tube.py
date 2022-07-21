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
Tests for ConcreteTubeSide model.

Author: Konica Mulani, Jaffer Ghouse
"""

import pytest
from pyomo.environ import (ConcreteModel, TerminationCondition,
                           SolverStatus, value, units as pyunits)

from idaes.core import FlowsheetBlock
from idaes.models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern

from idaes.generic_models.properties import iapws95

from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.solvers import get_solver
from idaes.core.util.testing import initialization_tester
from .heat_exchanger_tube import ConcreteTubeSide

# Get default solver for testing
solver = get_solver()


class TestConcreteTube(object):

    "Test for concrete tube side"

    @pytest.fixture(scope="class")
    def concrete_tube(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(default={"dynamic": False})

        m.fs.properties = iapws95.Iapws95ParameterBlock(default={
            "phase_presentation": iapws95.PhaseType.LG})

        m.fs.unit = ConcreteTubeSide(
            default={"property_package": m.fs.properties,
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

        return m

    @pytest.mark.unit
    def test_build(self, concrete_tube):

        assert len(concrete_tube.fs.unit.config) == 14
        assert concrete_tube.fs.unit.config.flow_type == \
            HeatExchangerFlowPattern.cocurrent
        assert concrete_tube.fs.unit.config.finite_elements == 20
        assert concrete_tube.fs.unit.config.collocation_points == 5

        assert hasattr(concrete_tube.fs.unit, "tube_area")
        assert hasattr(concrete_tube.fs.unit, "tube_length")
        assert hasattr(concrete_tube.fs.unit, "d_tube_outer")
        assert hasattr(concrete_tube.fs.unit, "d_tube_inner")
        assert hasattr(concrete_tube.fs.unit,
                       "tube_heat_transfer_coefficient")
        assert hasattr(concrete_tube.fs.unit, "temperature_wall")
        assert hasattr(concrete_tube.fs.unit, "tube_heat_transfer_eq")
        assert hasattr(concrete_tube.fs.unit, "area_calc_tube")

        assert degrees_of_freedom(concrete_tube) == 0

    @pytest.mark.component
    def test_initialize(self, concrete_tube):
        initialization_tester(concrete_tube)

    @pytest.mark.component
    def test_solve(self, concrete_tube):
        results = solver.solve(concrete_tube)

        # Check for optimal solution
        assert results.solver.termination_condition == \
            TerminationCondition.optimal
        assert results.solver.status == SolverStatus.ok

        assert pytest.approx(1, abs=1e-5) == \
            value(concrete_tube.fs.unit.tube_outlet.flow_mol[0])

        assert pytest.approx(55702.16, abs=1e0) == \
            value(concrete_tube.fs.unit.tube_outlet.enth_mol[0])

        assert pytest.approx(101325, abs=1e2) == \
            value(concrete_tube.fs.unit.tube_outlet.pressure[0])

        assert pytest.approx(581.80, abs=1e2) == \
            value(concrete_tube.fs.unit.tube.properties[0, 1].temperature)

    @pytest.mark.component
    def test_conservation(self, concrete_tube):
        assert abs(value(concrete_tube.fs.unit.tube_inlet.flow_mol[0] -
                         concrete_tube.fs.unit.tube_outlet.flow_mol[0])) \
            <= 1e-6

        tube_side = value(pyunits.convert(
                concrete_tube.fs.unit.tube_outlet.flow_mol[0]
                * (concrete_tube.fs.unit.tube.
                   properties[0, 1].enth_mol_phase['Liq']
                   - concrete_tube.fs.unit.tube.properties[0, 0].
                   enth_mol_phase['Liq']),
                to_units=pyunits.W))
        assert abs(tube_side) == pytest.approx(23497.05, rel=1e-3)
