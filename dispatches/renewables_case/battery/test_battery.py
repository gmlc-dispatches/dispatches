# Import objects from pyomo package
from pyomo.dae import ContinuousSet
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

from idaes.generic_models.unit_models.battery import BatteryStorage


# Create the ConcreteModel and the FlowsheetBlock, and attach the flowsheet block to it.
m = ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": True})   # dynamic or ss flowsheet needs to be specified here

# Add Hydrogen properties parameter block to the flowsheet with specifications
m.fs.properties = GenericParameterBlock(default=h2_config)

print("With just the property package, the DOF is {0}".format(degrees_of_freedom(m)))

# Add Battery Unit Model
m.fs.battery = BatteryStorage()
m.fs.battery.dt.set_value(1)
m.fs.battery.initial_state_of_charge.fix(0)
m.fs.battery.initial_energy_throughput.fix(0)
m.fs.battery.nameplate_power.fix(5)
m.fs.battery.nameplate_energy.fix(20)
m.fs.battery.elec_in[0].fix(5)
m.fs.battery.elec_in[1].fix(0)
m.fs.battery.elec_out[0].fix(0)
m.fs.battery.elec_out[1].fix(5)

print("Adding Battery, the DOF is {0}".format(degrees_of_freedom(m)))

assert_units_consistent(m)

solver = SolverFactory('ipopt')
solver.solve(m.fs)

n_ts = len(m.fs.config.time)
print("\n===Test 1===")


print("\nbattery inlet kW", [m.fs.battery.elec_in[t].value for t in range(n_ts)])
print("battery outlet kW", [m.fs.battery.elec_out[t].value for t in range(n_ts)])
print("battery soc", [m.fs.battery.state_of_charge[t].value for t in range(n_ts)])
print("battery energy throughput", [m.fs.battery.energy_throughput[t].value for t in range(n_ts)])

