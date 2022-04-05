#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
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
from dispatches.models.nuclear_case.unit_models.\
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
    m.fs.mixer = Mixer(
        default={"property_package": m.fs.h2turbine_props,
                 "inlet_list":
                 ["air_feed", "hydrogen_feed"]}
    )

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    m.fs.H2_mass = 2.016/1000

    m.fs.H2_production = Expression(
        expr=m.fs.pem.outlet.flow_mol[0] * m.fs.H2_mass)

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

    # add arcs
    m.fs.translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )

    # add arcs
    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    # expand arcs
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def set_inputs(m):
    # Hydrogen Production
    m.fs.nuclear_power = 100e3  # Input in kW.

    # Units are kW; Value here is to prove 54.517 kW makes 1 kg of H2 \
    # 54.517kW*hr/kg H2 based on H-tec systems
    m.fs.pem.electricity_in.electricity.fix(m.fs.nuclear_power)

    # Conversion of kW to mol/sec of H2. (elec*elec_to_mol) \
    # based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)

    # Base it on the TM2500 Aero-derivative Turbine.
    # Inlet Conditions of the inlet to the compressor.

    # Modified feed - only air flow, no hydrogen
    m.fs.mixer.air_feed.flow_mol[0].fix(2650)
    m.fs.mixer.air_feed.temperature[0].fix(288.15)
    m.fs.mixer.air_feed.pressure[0].fix(101325)

    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)

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

    # Initialize mixer
    propagate_state(m.fs.translator_to_mixer)
    m.fs.mixer.initialize()

    # Begin Initialization and solve for the system.
    propagate_state(m.fs.mixer_to_turbine)
    m.fs.h2_turbine.initialize()

    return m


if __name__ == "__main__":

    m = create_model()
    m = set_inputs(m)
    m = initialize_model(m)

    solver = SolverFactory('ipopt')
    res = solver.solve(m, tee=True)

    print("#### PEM ###")

    print("Hydrogen flow out of PEM (mol/sec)",
          value(m.fs.pem.outlet.flow_mol[0]))
    print("Hydrogen flow out of PEM (kg/sec)",
          value(m.fs.H2_production))
    print("Hydrogen flow out of PEM (kg/hr)",
          value(m.fs.H2_production) * 3600)

    print("#### Mixer ###")
    m.fs.mixer.report()

    print("#### Hydrogen Turbine ###")
    m.fs.h2_turbine.compressor.report()
    m.fs.h2_turbine.stoic_reactor.report()
    m.fs.h2_turbine.turbine.report()








