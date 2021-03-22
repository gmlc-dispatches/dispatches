
"""
Discharge heat exchanger model.

Heating source: thermal oil
Thermal material: steam
Author: Konor Frick and Jaffer Ghouse
Date: February 16, 2021
"""

import pytest
from pyomo.environ import ConcreteModel, SolverFactory, units, value

# Import IDAES components
from idaes.core import FlowsheetBlock

# Import heat exchanger unit model
from idaes.generic_models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern)
from idaes.core.util.model_statistics import degrees_of_freedom
# Import steam property package
from idaes.generic_models.properties.iapws95 import htpx, Iapws95ParameterBlock
from thermal_oil import ThermalOilParameterBlock

m = ConcreteModel()

m.fs = FlowsheetBlock(default={"dynamic": False})

m.fs.steam_prop = Iapws95ParameterBlock()
m.fs.therminol66_prop = ThermalOilParameterBlock()

m.fs.discharge_hx = HeatExchanger(
    default={"hot_side_name": "tube", "cold_side_name": "shell",
             "tube": {"property_package": m.fs.therminol66_prop},
             "shell": {"property_package": m.fs.steam_prop},
             "flow_pattern": HeatExchangerFlowPattern.countercurrent})

# Set inputs
#Steam
m.fs.discharge_hx.inlet_2.flow_mol[0].fix(4163)
m.fs.discharge_hx.inlet_2.pressure[0].fix(1.379e+6)
m.fs.discharge_hx.inlet_2.enth_mol[0].fix(htpx(T=300.15*units.K,
                                               P=1.379e+6*units.Pa))


#Thermal Oil
m.fs.discharge_hx.inlet_1.flow_mass[0].fix(833.3)
m.fs.discharge_hx.inlet_1.temperature[0].fix(256 + 273.15)
m.fs.discharge_hx.inlet_1.pressure[0].fix(101325)



#Designate the Area
m.fs.discharge_hx.area.fix(12180)
m.fs.discharge_hx.overall_heat_transfer_coefficient.fix(432.677)

#Designate the U Value.
print("Degrees of Freedom =", degrees_of_freedom(m))
m.fs.discharge_hx.initialize()
m.fs.discharge_hx.heat_duty.fix(1.066e+08)
m.fs.discharge_hx.area.unfix()



solver = SolverFactory("ipopt")
solver.solve(m, tee=True)
m.fs.discharge_hx.report()


#Tests to make sure the discharge cycle is functioning properly.
assert value(m.fs.discharge_hx.outlet_1.temperature[0]) == pytest.approx(473.5, rel=1e-1)
assert value(m.fs.discharge_hx.outlet_2.enth_mol[0]) == pytest.approx(27668.5, rel=1e-1)


