# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory, TransformationFactory
from pyomo.network import Arc

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

m.fs.windpower = Wind_Power()
# ((wind m/s, wind degrees from north clockwise, probability), )
# TODO add time series resource info
m.fs.windpower.config.resource_probability_density = ((1.5, 180, .12583),
                                                   (10, 180, .3933),
                                                   (15, 180, .18276),
                                                   (20, 180, .1341),
                                                   (24, 180, .14217),
                                                   (30, 180, .0211))
m.fs.windpower.system_capacity = 50000 # kW
m.fs.windpower.initialize()

print("Adding Wind, the DOF is {0}".format(degrees_of_freedom(m)))

m.fs.electrolyzer = PEM_Electrolyzer(default={"property_package": m.fs.properties})
m.fs.electrolyzer.electricity_to_mol.fix(5)
m.fs.electrolyzer.initialize()

# TODO check unit consistency

m.fs.connection = Arc(source=m.fs.windpower.outlet, dest=m.fs.electrolyzer.inlet)
TransformationFactory("network.expand_arcs").apply_to(m)


print("Adding PEM, the DOF is {0}".format(degrees_of_freedom(m)))


print("===Test 1===")
solver = SolverFactory('ipopt')
solver.solve(m.fs)

print("wind eff", m.fs.windpower.capacity_factor.value)
print("wind outlet kW", m.fs.windpower.outlet.electricity[0].value)

print("pem inlet kW:", m.fs.electrolyzer.inlet.electricity[0].value)
print("pem outlet H2 mols:", m.fs.electrolyzer.outlet.flow_mol[0].value)

print("pem outlet temp:", m.fs.electrolyzer.outlet.temperature[0].value)
print("pem outlet pres:", m.fs.electrolyzer.outlet.pressure[0].value)


# Test 2 fails with exceeding state_bounds in h2_ideal_vap-- how to work with this?
print("===Test 2===")
m.fs.windpower.system_capacity = 50000000000
solver.solve(m.fs)

print("wind eff", m.fs.windpower.capacity_factor.value)
print("wind outlet kW", m.fs.windpower.outlet.electricity[0].value)

print("pem inlet kW:", m.fs.electrolyzer.inlet.electricity[0].value)
print("pem outlet H2 mols:", m.fs.electrolyzer.outlet.flow_mol[0].value)

print("pem outlet temp:", m.fs.electrolyzer.outlet.temperature[0].value)
print("pem outlet pres:", m.fs.electrolyzer.outlet.pressure[0].value)

# from six import StringIO
# os = StringIO()
# m.pprint(ostream=os)
# print(os.getvalue())

