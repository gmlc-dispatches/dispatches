
"""
Discharge heat exchanger model.

Heating source: thermal oil
Thermal material: steam
"""


from pyomo.environ import ConcreteModel, SolverFactory, units

# Import IDAES components
from idaes.core import FlowsheetBlock

# Import heat exchanger unit model
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_lmtd_callback,
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
m.fs.discharge_hx.inlet_2.flow_mol[0].fix(5)
m.fs.discharge_hx.inlet_2.pressure[0].fix(1.379e+6)
m.fs.discharge_hx.inlet_2.enth_mol[0].fix(htpx(T=300.15*units.K,
                                               P=1.379e+6*units.Pa))


#Thermal Oil
m.fs.discharge_hx.inlet_1.flow_mass[0].fix(138.9)
m.fs.discharge_hx.inlet_1.temperature[0].fix(260 + 273.15)
m.fs.discharge_hx.inlet_1.pressure[0].fix(101325)



#Designate the Area
m.fs.discharge_hx.area.fix(12180*0.1)

#Designate the U Value.
m.fs.discharge_hx.overall_heat_transfer_coefficient.fix(0.1)
print("Degrees of Freedom =", degrees_of_freedom(m))
m.fs.discharge_hx.initialize()
solver = SolverFactory("ipopt")
solver.solve(m, tee=True)
m.fs.discharge_hx.report()

