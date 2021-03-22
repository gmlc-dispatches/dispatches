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
Charge heat exchanger model.

Heating source: steam
Thermal material: thermal oil
Author: Konor Frick and Jaffer Ghouse
Date: February 16, 2021
"""


# Import Pyomo libraries
import pytest
from pyomo.environ import ConcreteModel, SolverFactory, units, value

# Import IDAES components
from idaes.core import FlowsheetBlock

# Import heat exchanger unit model
from idaes.generic_models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern)

# Import steam property package
from idaes.generic_models.properties.iapws95 import htpx, Iapws95ParameterBlock
from thermal_oil import ThermalOilParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom

m = ConcreteModel()

m.fs = FlowsheetBlock(default={"dynamic": False})

m.fs.steam_prop = Iapws95ParameterBlock()
m.fs.therminol66_prop = ThermalOilParameterBlock()

m.fs.charge_hx = HeatExchanger(
    default={"shell": {"property_package": m.fs.steam_prop},
             "tube": {"property_package": m.fs.therminol66_prop},
             "flow_pattern": HeatExchangerFlowPattern.countercurrent})

# Set inputs
#Steam
m.fs.charge_hx.inlet_1.flow_mol[0].fix(4163)
m.fs.charge_hx.inlet_1.enth_mol[0].fix(htpx(T=573.15*units.K,
                                            P=5.0e+6*units.Pa))
m.fs.charge_hx.inlet_1.pressure[0].fix(5.0e+6)

#Thermal Oil
m.fs.charge_hx.inlet_2.flow_mass[0].fix(833.3)
m.fs.charge_hx.inlet_2.temperature[0].fix(200 + 273.15)
m.fs.charge_hx.inlet_2.pressure[0].fix(101325)

m.fs.charge_hx.area.fix(12180)
m.fs.charge_hx.overall_heat_transfer_coefficient.fix(432.677)

print("Degrees of Freedom =", degrees_of_freedom(m))



m.fs.charge_hx.initialize()
m.fs.charge_hx.heat_duty.fix(1.066e+08)
#Needed to make the system solve.
m.fs.charge_hx.overall_heat_transfer_coefficient.unfix()

print("Therminol specific heat", m.fs.charge_hx.inlet_2)
solver = SolverFactory("ipopt")
solver.solve(m, tee=True)
m.fs.charge_hx.report()


#Testing the exit values of the heat exchanger.

assert value(m.fs.charge_hx.outlet_2.temperature[0]) == pytest.approx(528.83, rel=1e-1)
assert value(m.fs.charge_hx.outlet_1.enth_mol[0]) == pytest.approx(27100.28, rel=1e-1)




