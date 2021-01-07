# Import objects from pyomo package
from pyomo.environ import ConcreteModel, SolverFactory

# Import the main FlowsheetBlock from IDAES. The flowsheet block will contain the unit model
from idaes.core import FlowsheetBlock

# Import the H2 property package to create a properties block for the flowsheet
from idaes.generic_models.properties.h2_ideal_vap import configuration
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

from idaes.generic_models.unit_models.pem_electrolyzer import PEM_Electrolyzer


# Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
m = ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": False})   # dynamic or ss flowsheet needs to be specified here

# Add properties parameter block to the flowsheet with specifications
m.fs.properties = GenericParameterBlock(default=configuration)

m.fs.electrolyzer = PEM_Electrolyzer(default={"property_package": m.fs.properties})
m.fs.electrolyzer.inlet.electricity.fix(1)
m.fs.electrolyzer.electricity_to_mol.fix(5)
m.fs.electrolyzer.initialize()

solver = SolverFactory('ipopt')
solver.solve(m.fs)

print("inlet electricity kW:", m.fs.electrolyzer.inlet.electricity[0].value)
print("outlet H2 mols:", m.fs.electrolyzer.outlet.flow_mol[0].value)

print("temp:", m.fs.electrolyzer.outlet.temperature[0].value)
print("pres:", m.fs.electrolyzer.outlet.pressure[0].value)

