#############################################################################
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
Nuclear Flowsheet
Author: Konor Frick
Date: April 20, 2021
"""

from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TransformationFactory,
                           value)
from pyomo.network import Arc, SequentialDecomposition
from pyomo.environ import Reference, Var, Reals, Constraint, Set, units as pyunits
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import (PressureChanger,
                                        Mixer,
                                        Separator as Splitter,
                                        Heater,
                                        StoichiometricReactor)

### Hydrogen Properties -----------------------------------------
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as configuration1
import dispatches.models.nuclear_case.properties.h2_reaction \
    as reaction_props
from dispatches.models.nuclear_case.Hydrogen_Turbine.\
    hydrogen_turbine_unit import HydrogenTurbine
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

from idaes.core.util.model_statistics import degrees_of_freedom

### PEM Properties -------------------------------------------------
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer

##Model construction
m = ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": False})  # dynamic or ss flowsheet needs to be specified here

#------------------------------------------------------------------------------------
# Hydrogen Production Portion--------------------------------------------------------
#------------------------------------------------------------------------------------
m.fs.nuclear_electrical = 1000e3 # Input in kW.

# Add properties parameter block to the flowsheet with specifications
m.fs.PEM_properties = GenericParameterBlock(default=configuration)

m.fs.unit = PEM_Electrolyzer(default={"property_package": m.fs.PEM_properties})

m.fs.unit.electricity_in.electricity.fix(m.fs.nuclear_electrical)    ## Units are kW; Value here is to prove 54.517 kW makes 1 kg of H2 \
                                                    # 54.517kW*hr/kg H2 based on H-tec systems

m.fs.unit.electricity_to_mol.fix(0.002527406)       ## Conversion of kW to mol/sec of H2. (elec*elec_to_mol) \
                                                    # based on H-tec design of 54.517kW-hr/kg

m.fs.unit.initialize()


solver = SolverFactory('ipopt')
results = solver.solve(m.fs, tee=True)
print("Hydrogen flow out of PEM (mol/sec)" ,m.fs.unit.outlet.flow_mol[0].value)

m.fs.H2_mass = 2.016/1000

m.fs.H2_production = Expression(expr=m.fs.unit.outlet.flow_mol[0].value * m.fs.H2_mass)
print("Hydrogen flow out of PEM (kg/sec)", m.fs.H2_production.expr)
print("Hydrogen flow out of PEM (kg/hr)", m.fs.H2_production.expr * 3600)
#m.fs.unit.report()
#print(degrees_of_freedom(m))

m.fs.unit.outlet.display()
#------------------------------------------------------------------------------------
# Hydrogen Turbine Peaking Portion --------------------------------------------------
#------------------------------------------------------------------------------------

#Base it on the TM2500 Aero-derivative Turbine.

m.fs.h2turbine_props = GenericParameterBlock(default=configuration1)

#m.fs.Mixer = Mixer(default={"property_package":m.fs.h2turbine_props,
#                            "inlet_list", ["Air", "pure_hydrogen"]})



# Air Properties


m.fs.reaction_params = reaction_props. \
    H2ReactionParameterBlock(default={"property_package": m.fs.h2turbine_props})

m.fs.h2_turbine = HydrogenTurbine(default={"property_package": m.fs.h2turbine_props,
                         "reaction_package": m.fs.reaction_params})



# Inlet Conditions of the inlet to the compressor.
m.fs.h2_turbine.compressor.inlet.flow_mol[0].fix(4135.2)
m.fs.h2_turbine.compressor.inlet.temperature[0].fix(288.15)
m.fs.h2_turbine.compressor.inlet.pressure[0].fix(101325)

m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "oxygen"].fix(0.188)
m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "argon"].fix(0.003)
m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "nitrogen"].fix(0.702)
m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "water"].fix(0.022)
m.fs.h2_turbine.compressor.inlet.mole_frac_comp[0, "hydrogen"].fix(0.085)

m.fs.h2_turbine.compressor.deltaP.fix(2.401e6)
m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

# Specify the Stoichiometric Conversion Rate of hydrogen
# in the equation shown below
# H2(g) + O2(g) --> H2O(g) + energy
# Complete Combustion
m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

# Turbine Parameters
m.fs.h2_turbine.turbine.deltaP.fix(-2.401e6)
m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

print(degrees_of_freedom(m))
assert degrees_of_freedom(m) == 0

solver = SolverFactory("ipopt")

# Begin Initialization and solve for the system.
m.fs.h2_turbine.initialize()

solver.solve(m, tee=True)