# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, TransformationFactory
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock

from idaes.core.util.model_statistics import degrees_of_freedom, unfixed_variables_in_activated_equalities_set, activated_equalities_set

# Import the H2 property package to create a properties block for the flowsheet
from idaes.generic_models.properties.h2_ideal_vap import configuration as h2_config
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

from idaes.generic_models.unit_models.pem_electrolyzer import PEM_Electrolyzer
from idaes.generic_models.unit_models.wind_power import Wind_Power


# Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
m = ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": True})   # dynamic or ss flowsheet needs to be specified here

# Add properties parameter block to the flowsheet with specifications
m.fs.properties = GenericParameterBlock(default=h2_config)

print("With just the property package, the DOF is {0}".format(degrees_of_freedom(m)))

# ((wind m/s, wind degrees from north clockwise, probability), )
resource_timeseries = dict()
for time in list(m.fs.config.time.data()):
    resource_timeseries[time] = ((10, 180, 0.5),
                                 (24, 180, 0.5))

wind_config = {'resource_probability_density': resource_timeseries}

m.fs.windpower = Wind_Power(default=wind_config)
m.fs.windpower.system_capacity.fix(50000) # kW

print("Adding Wind, the DOF is {0}".format(degrees_of_freedom(m)))

m.fs.electrolyzer = PEM_Electrolyzer(default={"property_package": m.fs.properties})
m.fs.electrolyzer.electricity_to_mol.fix(5)
m.fs.electrolyzer.initialize()

m.fs.connection = Arc(source=m.fs.windpower.power_out, dest=m.fs.electrolyzer.inlet)
TransformationFactory("network.expand_arcs").apply_to(m)

assert_units_consistent(m)

print("Adding PEM, the DOF is {0}".format(degrees_of_freedom(m)))

print("===Test 1===")
solver = SolverFactory('ipopt')
solver.solve(m.fs)

print("wind eff", m.fs.windpower.capacity_factor[0].value)
print("wind outlet kW", m.fs.windpower.power_out.electricity[0].value)

print("pem inlet kW:", m.fs.electrolyzer.inlet.electricity[0].value)
print("pem outlet H2 mols:", m.fs.electrolyzer.outlet.flow_mol[0].value)

print("pem outlet temp:", m.fs.electrolyzer.outlet.temperature[0].value)
print("pem outlet pres:", m.fs.electrolyzer.outlet.pressure[0].value)

# Test 2 fails with exceeding state_bounds in h2_ideal_vap-- how to work with this?
print("===Test 2===")
m.fs.windpower.system_capacity.set_value(50000000000)
solver.solve(m.fs)

print("wind eff", m.fs.windpower.capacity_factor[0].value)
print("wind outlet kW", m.fs.windpower.power_out.electricity[0].value)

print("pem inlet kW:", m.fs.electrolyzer.inlet.electricity[0].value)
print("pem outlet H2 mols:", m.fs.electrolyzer.outlet.flow_mol[0].value)

print("pem outlet temp:", m.fs.electrolyzer.outlet.temperature[0].value)
print("pem outlet pres:", m.fs.electrolyzer.outlet.pressure[0].value)
