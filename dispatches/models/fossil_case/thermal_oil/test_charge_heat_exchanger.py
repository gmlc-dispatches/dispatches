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
Charge heat exchanger model.

Heating source: steam
Thermal material: thermal oil
Author: Konor Frick and Jaffer Ghouse
Date: February 16, 2021
"""


# Import Pyomo libraries
import pytest
from pyomo.environ import ConcreteModel, SolverFactory, units, value, \
    TerminationCondition, SolverStatus

# Import IDAES components
from idaes.core import FlowsheetBlock

# Import heat exchanger unit model
from idaes.models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern)

# Import steam property package
from idaes.generic_models.properties.iapws95 import htpx, Iapws95ParameterBlock
from dispatches.models.fossil_case.thermal_oil.thermal_oil \
    import ThermalOilParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver


def test_charge():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    m.fs.steam_prop = Iapws95ParameterBlock()
    m.fs.therminol66_prop = ThermalOilParameterBlock()

    m.fs.charge_hx = HeatExchanger(
        default={"shell": {"property_package": m.fs.steam_prop},
                 "tube": {"property_package": m.fs.therminol66_prop},
                 "flow_pattern": HeatExchangerFlowPattern.countercurrent})

    # Set inputs
    # Steam
    m.fs.charge_hx.inlet_1.flow_mol[0].fix(4163)
    m.fs.charge_hx.inlet_1.enth_mol[0].fix(htpx(T=573.15*units.K,
                                                P=5.0e+6*units.Pa))
    m.fs.charge_hx.inlet_1.pressure[0].fix(5.0e+6)

    # Thermal Oil
    m.fs.charge_hx.inlet_2.flow_mass[0].fix(833.3)
    m.fs.charge_hx.inlet_2.temperature[0].fix(200 + 273.15)
    m.fs.charge_hx.inlet_2.pressure[0].fix(101325)

    m.fs.charge_hx.area.fix(12180)
    m.fs.charge_hx.overall_heat_transfer_coefficient.fix(432.677)

    m.fs.charge_hx.initialize()
    m.fs.charge_hx.heat_duty.fix(1.066e+08)

    # Needed to make the system solve.
    m.fs.charge_hx.overall_heat_transfer_coefficient.unfix()

    solver = get_solver()
    results = solver.solve(m)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    # Testing the exit values of the heat exchanger.
    assert value(m.fs.charge_hx.outlet_2.temperature[0]) == \
        pytest.approx(528.83, rel=1e-1)
    assert value(m.fs.charge_hx.outlet_1.enth_mol[0]) == \
        pytest.approx(27100.28, rel=1e-1)




