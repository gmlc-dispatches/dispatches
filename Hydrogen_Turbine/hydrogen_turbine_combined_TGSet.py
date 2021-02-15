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


from pyomo.environ import ConcreteModel, SolverFactory, units, Constraint, Var, Expression, TransformationFactory
from idaes.core import (FlowsheetBlock,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        MomentumBalanceType)
from pyomo.network import Arc, SequentialDecomposition
from idaes.generic_models.unit_models import (Compressor, StoichiometricReactor, Turbine)


from task_2.Hydrogen_Turbine.hturbine_ideal_vap import configuration
import task_2.Hydrogen_Turbine.h2_reaction as reaction_props
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock


from idaes.core.util.model_statistics import degrees_of_freedom

m = ConcreteModel()

m.fs = FlowsheetBlock(default={"dynamic": False})

m.fs.properties1 = GenericParameterBlock(default=configuration)  #Air Properties
m.fs.unit = Compressor(default={"property_package": m.fs.properties1})

#print("Degrees of Freedom =", degrees_of_freedom(m))


m.fs.reaction_params = reaction_props.H2ReactionParameterBlock(default={"property_package": m.fs.properties1})

#Adding Stoichiometric Reactor Model
m.fs.R101 = StoichiometricReactor(
            default={"property_package": m.fs.properties1,
                     "reaction_package": m.fs.reaction_params,
                     "has_heat_of_reaction": True,
                     "has_heat_transfer": False,
                     "has_pressure_change": False})


#Turbine

m.fs.T1000 = Turbine(default={"property_package": m.fs.properties1})


#Creation of Connections between the three separate pieces.
m.fs.s01 = Arc(source=m.fs.unit.outlet, destination=m.fs.R101.inlet)
m.fs.s02 = Arc(source=m.fs.R101.outlet, destination=m.fs.T1000.inlet)

TransformationFactory("network.expand_arcs").apply_to(m)

print(degrees_of_freedom(m))

#Initial Conditions***********************************************************
#*****************************************************************************
#Initial Conditions of the inlet to the compressor.
m.fs.unit.inlet.flow_mol[0].fix(4146)
m.fs.unit.inlet.temperature[0].fix(300)
m.fs.unit.inlet.pressure[0].fix(101325)


m.fs.unit.inlet.mole_frac_comp[0, "oxygen"].fix(0.188)
m.fs.unit.inlet.mole_frac_comp[0, "argon"].fix(0.003)
m.fs.unit.inlet.mole_frac_comp[0, "nitrogen"].fix(0.70)
m.fs.unit.inlet.mole_frac_comp[0, "water"].fix(0.022)
m.fs.unit.inlet.mole_frac_comp[0, "hydrogen"].fix(0.087)

m.fs.unit.deltaP.fix(2.401e6)
m.fs.unit.efficiency_isentropic.fix(0.86)


#Combustion Chamber Values
m.fs.R101.conversion = Var(initialize=0.75, bounds=(0, 1))

m.fs.R101.conv_constraint = Constraint(
    expr=m.fs.R101.conversion*m.fs.R101.inlet.
    mole_frac_comp[0, "hydrogen"] ==
    (m.fs.R101.inlet.mole_frac_comp[0, "hydrogen"] -
     m.fs.R101.outlet.mole_frac_comp[0, "hydrogen"]))

#  Specify the Stoichiometric Conversion Rate of hydrogen in the equation shown below
#  H2(g) + O2(g) --> H2O(g) + energy
#  Complete Combustion
m.fs.R101.conversion.fix(1.0)

#Turbine Parameters
m.fs.T1000.deltaP.fix(-2.401e6)
m.fs.T1000.efficiency_isentropic.fix(0.89)


print("Degrees of Freedom =", degrees_of_freedom(m))



#Begin Initialization and solve for the system.

solver = SolverFactory("ipopt")
m.fs.unit.initialize()
solver.solve(m, tee=True)
solver = SolverFactory("ipopt")
m.fs.R101.initialize()
solver.solve(m, tee=True)
m.fs.T1000.initialize()
solver = SolverFactory("ipopt")
#solver.options = {'tol': 1e-6, 'max_iter': 5000}
solver.solve(m, tee=True)


#m.fs.display()
m.fs.unit.report()
m.fs.R101.report()
m.fs.T1000.report()

