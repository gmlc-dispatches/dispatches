#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################

__author__ = "Konor Frick, Radhakrishna Tumbalam Gooty"

"""
This file contains a function to construct the generic nuclear flowsheet, and 
a function to fix the degrees of freedom and initialize.

To simulate the flowsheet, use:
# Create a concrete model
mdl = ConcreteModel()

# Build the nuclear flowsheet
build_ne_flowsheet(mdl)

# Fix the degrees of freedom and initialize the flowsheet
fix_dof_and_initialize(mdl)

# Ensure that the number of degrees of freedom is zero
print("Degrees of freedom: ", degrees_of_freedom(mdl))
assert degrees_of_freedom(mdl) == 0

# Simulate the entire flowsheet
res = get_solver().solve(mdl, tee=True)

see multiperiod_design_pricetaker.ipynb for more details
"""
# Pyomo imports
from pyomo.environ import (Constraint,
                           ConcreteModel,
                           TransformationFactory)
from pyomo.network import Arc

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.models.unit_models import (Translator,
                                              Mixer,
                                              MomentumMixingType)
from idaes.models.properties.modular_properties.base.generic_property \
    import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers import get_solver

# DISPATCHES imports
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config
import dispatches.models.nuclear_case.properties.h2_reaction \
    as h2_reaction_props
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter
from dispatches.models.nuclear_case.unit_models.hydrogen_tank_simplified import SimpleHydrogenTank
from dispatches.models.nuclear_case.unit_models.hydrogen_turbine_unit import HydrogenTurbine


def build_ne_flowsheet(m, **kwargs):
    """
    This function builds the entire nuclear flowsheet by adding the
    required models and arcs connecting the models.
    """
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Load thermodynamic and reaction packages
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)
    m.fs.h2turbine_props = GenericParameterBlock(default=hturbine_config)
    m.fs.reaction_params = h2_reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    # Add electrical splitter
    m.fs.np_power_split = ElectricalSplitter(default={
        "num_outlets": 2,
        "outlet_list": ["np_to_grid", "np_to_pem"],
        "add_split_fraction_vars": True})

    # Add PEM electrolyzer
    m.fs.pem = PEM_Electrolyzer(default={
        "property_package": m.fs.h2ideal_props})

    # Add hydrogen tank
    m.fs.h2_tank = SimpleHydrogenTank(default={
        "property_package": m.fs.h2ideal_props})

    # Add translator block
    m.fs.translator = Translator(default={
        "inlet_property_package": m.fs.h2ideal_props,
        "outlet_property_package": m.fs.h2turbine_props})

    # Add translator block constraints
    m.fs.translator.eq_flow_hydrogen = Constraint(
        expr=m.fs.translator.inlet.flow_mol[0] ==
             m.fs.translator.outlet.flow_mol[0])

    m.fs.translator.eq_temperature = Constraint(
        expr=m.fs.translator.inlet.temperature[0] ==
             m.fs.translator.outlet.temperature[0])

    m.fs.translator.eq_pressure = Constraint(
        expr=m.fs.translator.inlet.pressure[0] ==
             m.fs.translator.outlet.pressure[0])

    # Add mixer block
    # using minimize pressure for all inlets and outlet of the mixer
    # because pressure of inlets is already fixed in flowsheet,
    # using equality will over-constrain
    m.fs.mixer = Mixer(default={
        "momentum_mixing_type": MomentumMixingType.minimize,
        "property_package": m.fs.h2turbine_props,
        "inlet_list": ["air_feed", "hydrogen_feed"]})

    # Add hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    """
    Connect the individual blocks via Arcs
    """
    # Connect the electrical splitter and PEM
    m.fs.arc_np_to_pem = Arc(
        source=m.fs.np_power_split.np_to_pem_port,
        destination=m.fs.pem.electricity_in
    )

    # Connect the pem electrolyzer and h2 tank
    m.fs.arc_pem_to_h2_tank = Arc(
        source=m.fs.pem.outlet,
        destination=m.fs.h2_tank.inlet
    )

    # Connect h2 tank and translator
    m.fs.arc_h2_tank_to_translator = Arc(
        source=m.fs.h2_tank.outlet_to_turbine,
        destination=m.fs.translator.inlet
    )

    # Connect translator and mixer
    m.fs.arc_translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )

    # Connect mixer and h2 turbine
    m.fs.arc_mixer_to_h2_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def fix_dof_and_initialize(m, **kwargs):
    """
    This function fixes the degrees of freedom of each unit and initializes it
    np_power_output       : Power output from nuclear power plant in kW
    pem_outlet_pressure   : Outlet pressure of hydrogen from PEM in bar
    pem_outlet_temperature: Outlet temperature of hydrogen from PEM in K
    air_h2_ratio          : Ratio of molar flowrate of air to molar flowrate of hydrogen
                            entering the hydrogen turbine
    compressor_dp         : Pressure difference (in bar) between the outlet and the inlet of
                            the hydrogen turbine's compressor. The same pressure difference
                            is used for hydrogen turbine's turbine.
    """
    options = kwargs.get("options", {})
    np_power_output = options.get("np_power_output", 1000 * 1e3)
    pem_outlet_pressure = options.get("pem_outlet_pressure", 1.01325)
    pem_outlet_temperature = options.get("pem_outlet_temperature", 300)
    air_h2_ratio = options.get("air_h2_ratio", 10.76)
    compressor_dp = options.get("compressor_dp", 24.01)

    # Fix the dof of the electrical splitter and initialize
    m.fs.np_power_split.electricity[0].fix(np_power_output)  # in kW
    m.fs.np_power_split.split_fraction["np_to_grid", 0].fix(0.8)

    m.fs.np_power_split.initialize()

    # Fix the dof of the electrolyzer and initialize
    # Conversion of kW to mol/sec of H2 based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)
    m.fs.pem.outlet.pressure.fix(pem_outlet_pressure * 1e5)
    m.fs.pem.outlet.temperature.fix(pem_outlet_temperature)

    propagate_state(m.fs.arc_np_to_pem)
    m.fs.pem.initialize()

    # Fix the dof of the tank and initialize
    m.fs.h2_tank.dt.fix(3600)
    m.fs.h2_tank.tank_holdup_previous.fix(0)
    m.fs.h2_tank.outlet_to_turbine.flow_mol.fix(10)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.fix(10)
    m.fs.h2_tank.outlet_to_turbine.mole_frac_comp[0, "hydrogen"].fix(1)
    m.fs.h2_tank.outlet_to_pipeline.mole_frac_comp[0, "hydrogen"].fix(1)

    propagate_state(m.fs.arc_pem_to_h2_tank)
    m.fs.h2_tank.initialize()

    # Fix the dof of the translator block and initialize
    m.fs.translator.outlet.mole_frac_comp[0, "hydrogen"].fix(0.99)
    m.fs.translator.outlet.mole_frac_comp[0, "oxygen"].fix(0.01 / 4)
    m.fs.translator.outlet.mole_frac_comp[0, "argon"].fix(0.01 / 4)
    m.fs.translator.outlet.mole_frac_comp[0, "nitrogen"].fix(0.01 / 4)
    m.fs.translator.outlet.mole_frac_comp[0, "water"].fix(0.01 / 4)

    propagate_state(m.fs.arc_h2_tank_to_translator)
    m.fs.translator.initialize()

    # Fix the degrees of freedom of mixer and initialize
    m.fs.mixer.air_feed.flow_mol[0].fix(
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].value * air_h2_ratio
    )
    m.fs.mixer.air_feed.temperature[0].fix(pem_outlet_temperature)
    m.fs.mixer.air_feed.pressure[0].fix(pem_outlet_pressure * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)

    propagate_state(m.fs.arc_translator_to_mixer)
    m.fs.mixer.initialize()

    # Fix the degrees of freedom of H2 turbine and initialize
    m.fs.h2_turbine.compressor.deltaP.fix(compressor_dp * 1e5)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)
    m.fs.h2_turbine.turbine.deltaP.fix(-compressor_dp * 1e5)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    propagate_state(m.fs.arc_mixer_to_h2_turbine)
    m.fs.h2_turbine.initialize()

    return









