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
"""
# Pyomo imports
from pyomo.environ import (ConcreteModel,
                           Expression,
                           TransformationFactory)
from pyomo.network import Arc

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property \
    import GenericParameterBlock
from idaes.core.util.initialization import propagate_state
import idaes.logger as idaeslog

# DISPATCHES imports
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter
from dispatches.models.nuclear_case.unit_models.hydrogen_tank_simplified import SimpleHydrogenTank


def build_ne_flowsheet(np_power_production=500,
                       pem_capacity=100,
                       tank_capacity=5000,
                       h2_demand=0.35,
                       h2_price=4,
                       demand_type="variable"):
    """
    This function builds the entire nuclear flowsheet by adding the
    required models and arcs connecting the models.

    Args:
        np_power_production: Power output from the nuclear power plant (in MW)
        pem_capacity: Maximum power input to the PEM electrolyzer (in MW)
        tank_capacity: Maximum amount hydrogen that can be stored in the tank (in kg)
        h2_demand: Hydrogen demand (in kg/s)
        h2_price: Price of hydrogen (in $/kg)
        demand_type: "fixed"/"variable" hydrogen demand

    Returns:
        m: Object containing the nuclear flowsheet
    """
    mw_h2 = 2.016 * 1e-3  # Molecular mass of hydrogen in kg/mol

    # Define conversion factors
    # TODO: Use pyunits convert instead of defining conversion factors
    MW_to_kW = 1000
    kW_to_MW = 1e-3
    hours_to_s = 3600

    # TODO: Need to find a better way to import costing data.
    # Using a VOM cost of $2.3 per MWh for the nuclear power plant
    # Using a VOM cost of $1.3 per MWh for the PEM electrolyzer
    # Fixed O&M cost for the nuclear power plant is $120,000 per MW-year
    # Fixed O&M cost for the PEM electrolyzer is $47,900 per MW-year
    # Normalized FOM = (120,000 / 8760) = $13.7 per MWh
    # Normalized FOM = (47,900 / 8760) = $5.47 per MWh
    npp_fom = 13.7
    npp_vom = 2.3
    pem_fom = 5.47
    pem_vom = 1.3
    tank_vom = 0.01

    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Load thermodynamic and reaction packages
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

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

    TransformationFactory("network.expand_arcs").apply_to(m)

    # Initialize model
    fix_dof_and_initialize(m, np_power_production=np_power_production)

    # Unfix degrees of freedom for optimization
    unfix_dof(m)

    # Capacity constraints: Set upper bounds on capacities
    m.fs.pem.electricity.setub(pem_capacity * MW_to_kW)
    m.fs.h2_tank.tank_holdup.setub(tank_capacity / mw_h2)
    m.fs.h2_tank.tank_holdup_previous.setub(tank_capacity / mw_h2)

    # Hydrogen demand constraint
    if demand_type == "variable":
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].setub(h2_demand / mw_h2)

    elif demand_type == "fixed":
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix(h2_demand / mw_h2)

    # Treating the revenue generated from hydrogen as negative cost
    # To avoid degeneracy, we are adding an operating cost for hydrogen storage
    # FIXME: How do we include the FOM contribution?
    m.fs.operating_cost = Expression(
        expr=m.fs.np_power_split.electricity[0] * kW_to_MW * npp_vom +
             m.fs.pem.electricity[0] * kW_to_MW * pem_vom +
             m.fs.h2_tank.tank_holdup[0] * mw_h2 * tank_vom -
             m.fs.h2_tank.outlet_to_pipeline.flow_mol[0] * mw_h2 * hours_to_s * h2_price)

    return m


def fix_dof_and_initialize(m,
                           np_power_production=500,
                           pem_outlet_pressure=1.01325,
                           pem_outlet_temperature=300):
    """
    This function fixes the degrees of freedom of each unit and initializes it

    Args:
        m: Object containing the integrated nuclear plant flowsheet
        np_power_output: Power output from nuclear power plant in MW
        pem_outlet_pressure: Outlet pressure of hydrogen from PEM in bar
        pem_outlet_temperature: Outlet temperature of hydrogen from PEM in K

    Returns:
        None
    """
    # Fix the dof of the electrical splitter and initialize
    m.fs.np_power_split.electricity[0].fix(np_power_production * 1000)  # in kW
    m.fs.np_power_split.split_fraction["np_to_grid", 0].fix(0.95)

    m.fs.np_power_split.initialize(outlvl=idaeslog.WARNING)

    # Fix the dof of the electrolyzer and initialize
    # Conversion of kW to mol/sec of H2 based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)
    m.fs.pem.outlet.pressure.fix(pem_outlet_pressure * 1e5)
    m.fs.pem.outlet.temperature.fix(pem_outlet_temperature)

    propagate_state(m.fs.arc_np_to_pem)
    m.fs.pem.initialize(outlvl=idaeslog.WARNING)

    # Fix the dof of the tank and initialize
    m.fs.h2_tank.dt.fix(3600)
    m.fs.h2_tank.tank_holdup_previous.fix(0)
    m.fs.h2_tank.outlet_to_turbine.flow_mol.fix(0)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.fix(1)
    m.fs.h2_tank.outlet_to_turbine.mole_frac_comp[0, "hydrogen"].fix(1)
    m.fs.h2_tank.outlet_to_pipeline.mole_frac_comp[0, "hydrogen"].fix(1)

    propagate_state(m.fs.arc_pem_to_h2_tank)
    m.fs.h2_tank.initialize(outlvl=idaeslog.WARNING)

    return


def unfix_dof(m):
    """
    This function unfixes a few degrees of freedom for optimization

    Args:
        m: object containing the integrated nuclear plant flowsheet

    Returns:
        None
    """
    # Unfix split fractions in the power splitter
    m.fs.np_power_split.split_fraction.unfix()

    # Unfix the holdup_previous and outflow variables
    m.fs.h2_tank.tank_holdup_previous.unfix()
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.unfix()

    return
