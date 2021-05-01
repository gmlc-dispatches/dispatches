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
from idaes.core import FlowsheetBlock

from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as configuration1
import dispatches.models.nuclear_case.properties.h2_reaction \
    as reaction_props
from dispatches.models.nuclear_case.Hydrogen_Turbine.\
    hydrogen_turbine_unit import HydrogenTurbine
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.generic_models.unit_models import Translator, Mixer
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state

from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer


def create_model():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Add properties for PEM
    m.fs.PEM_properties = GenericParameterBlock(default=configuration)

    # Add properties for hydrogen turbine
    m.fs.h2turbine_props = GenericParameterBlock(default=configuration1)

    m.fs.reaction_params = reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    # Add the PEM electrolyzer unit
    m.fs.pem = PEM_Electrolyzer(
        default={"property_package": m.fs.PEM_properties})

    # Add translator block
    m.fs.translator = Translator(
        default={"inlet_property_package": m.fs.PEM_properties,
                 "outlet_property_package": m.fs.h2turbine_props})

    # Add mixer block
    # m.fs.mixer = Mixer(
    #     default={"property_package": m.fs.h2turbine_props,
    #              "inlet_list":
    #              ["air_feed", "hydrogen_feed"]}
    # )

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    m.fs.H2_mass = 2.016/1000

    m.fs.H2_production = Expression(
        expr=m.fs.pem.outlet.flow_mol[0].value * m.fs.H2_mass)

    # Add translator constraints
    # Set hydrogen flow and mole frac
    m.fs.translator.eq_flow_hydrogen = Constraint(
        expr=m.fs.translator.inlet.flow_mol[0] ==
        m.fs.translator.outlet.flow_mol[0]
    )

    m.fs.translator.mole_frac_hydrogen = Constraint(
        expr=m.fs.translator.outlet.mole_frac_comp[0, "hydrogen"] == 0.99
    )

    m.fs.translator.eq_temperature = Constraint(
        expr=m.fs.translator.inlet.temperature[0] ==
        m.fs.translator.outlet.temperature[0]
    )

    m.fs.translator.eq_pressure = Constraint(
        expr=m.fs.translator.inlet.pressure[0] ==
        m.fs.translator.outlet.pressure[0]
    )

    m.fs.translator.outlet.mole_frac_comp[0, "oxygen"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "argon"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "nitrogen"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "water"].fix(0.01/4)

    # add arcs
    m.fs.pem_to_translator = Arc(
        source=m.fs.pem.outlet,
        destination=m.fs.translator.inlet
    )

    # expand arcs
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def set_inputs(m):
    # Hydrogen Production
    m.fs.nuclear_power = 1000e3  # Input in kW.

    # Units are kW; Value here is to prove 54.517 kW makes 1 kg of H2 \
    # 54.517kW*hr/kg H2 based on H-tec systems
    m.fs.pem.electricity_in.electricity.fix(m.fs.nuclear_power)

    # Conversion of kW to mol/sec of H2. (elec*elec_to_mol) \
    # based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)

    # Base it on the TM2500 Aero-derivative Turbine.
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

    return m


def initialize_model(m):

    m.fs.pem.initialize()

    propagate_state(m.fs.pem_to_translator)
    m.fs.translator.initialize()

    # Begin Initialization and solve for the system.
    m.fs.h2_turbine.initialize()


    return m


m = create_model()
m = set_inputs(m)
m = initialize_model(m)

print("Hydrogen flow out of PEM (mol/sec)",
      m.fs.pem.outlet.flow_mol[0].value)
print("Hydrogen flow out of PEM (kg/sec)", m.fs.H2_production.expr)
print("Hydrogen flow out of PEM (kg/hr)", m.fs.H2_production.expr * 3600)
