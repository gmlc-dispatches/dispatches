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
"""
Renewable Energy Flowsheet
Author: Darice Guittet
Date: June 7, 2021
"""

import matplotlib.pyplot as plt
from functools import partial
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Param,
                           units as pyunits,
                           SolverFactory,
                           TransformationFactory,
                           NonNegativeReals,
                           Reference,
                           value)
from pyomo.network import Arc, Port
import idaes.core.util.scaling as iscale

import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.core.util.initialization import propagate_state
from idaes.models.properties.modular_properties.base.generic_property \
    import GenericParameterBlock
from idaes.models.unit_models import (Translator,
                                              Mixer,
                                              MomentumMixingType,
                                              Valve,
                                              ValveFunctionType)

from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config
import dispatches.models.nuclear_case.properties.h2_reaction \
    as h2_reaction_props

from idaes.models.unit_models.separator import Separator
from dispatches.models.nuclear_case.unit_models.hydrogen_turbine_unit import HydrogenTurbine
from dispatches.models.nuclear_case.unit_models.hydrogen_tank import HydrogenTank
from dispatches.models.nuclear_case.unit_models.hydrogen_tank_simplified import SimpleHydrogenTank as SimpleHydrogenTank
from dispatches.models.renewables_case.load_parameters import *
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter
from dispatches.models.renewables_case.battery import BatteryStorage
from dispatches.models.renewables_case.wind_power import Wind_Power
import matplotlib as mpl
mpl.use(mpl.rcParams['backend'])


def add_wind(m, wind_mw, wind_resource_config=None):
    """
    Adds a wind unit to the flowsheet with a fixed system capacity and that uses resource data to determine the capacity factors

    The resource data can be provided by configuring either `resource_speed` or `resource_probability_density`.
    The `resource_speed` is a list of wind speeds (m/s) per timestep of the flowsheet.
    The `resource_probability_density` is formatted as a 2d list where each row contains: (wind m/s, wind degrees from north clockwise, probability) 
    and the probabilities in all rows add up to 1.

    Args:
        m: existing ConcreteModel with a flowsheet `fs`
        wind_mw: capacity of wind model to be fixed
        wind_resource_config: dictionary of Windpower Config keys (`resource_speed`, `resource_probability_density`) and ConfigValues
    Returns:
        wind unit model in the flowsheet
    """
    m.fs.windpower = Wind_Power(default=wind_resource_config)
    m.fs.windpower.system_capacity.fix(wind_mw * 1e3)   # kW
    return m.fs.windpower


def add_pem(m, outlet_pressure_bar):
    """
    Adds a PEM electrolyzer unit to the flowsheet with a fixed outlet pressure and conversion rate. The properties package is h2_ideal_vap.

    The operating temperature and maximum pressure of the PEM are pulled from load_parameters.py

    Args:
        m: existing ConcreteModel with a flowsheet `fs`
        outlet_pressure_bar: operating pressure of the PEM
    Returns:
        PEM electrolyzer unit model in the flowsheet
        h2_ideal_vap property package in the flowsheet

    """
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)
    m.fs.h2ideal_props.set_default_scaling('flow_mol_phase', 1)
    m.fs.h2ideal_props.set_default_scaling('mole_frac_comp', 1)
    m.fs.h2ideal_props.set_default_scaling('mole_frac_phase_comp', 1)
    m.fs.h2ideal_props.set_default_scaling('flow_mol', 1)
    m.fs.h2ideal_props.set_default_scaling('enth_mol_phase', 0.1)

    m.fs.pem = PEM_Electrolyzer(
        default={"property_package": m.fs.h2ideal_props})

    # Conversion of kW to mol/sec of H2. (elec*elec_to_mol) based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)
    m.fs.pem.outlet.pressure.setub(max_pressure_bar * 1e5)
    m.fs.pem.outlet.pressure.fix(outlet_pressure_bar * 1e5)
    m.fs.pem.outlet.temperature.fix(pem_temp)
    return m.fs.pem, m.fs.h2ideal_props


def add_battery(m, batt_mw):
    """
    Adds a Battery unit to the flowsheet with a fixed power and duration of 4 hours.
    
    The charging and discharging efficiencies are set at 98%.

    Args:
        m: existing ConcreteModel with a flowsheet `fs`
        batt_mw: nameplate power
    Returns:
        battery unit model in flowsheet
    """
    m.fs.battery = BatteryStorage()
    m.fs.battery.charging_eta.set_value(0.95)
    m.fs.battery.discharging_eta.set_value(0.95)
    m.fs.battery.dt.set_value(timestep_hrs)
    m.fs.battery.nameplate_power.fix(batt_mw * 1e3)
    m.fs.battery.duration = Param(default=4, mutable=True, units=pyunits.kWh/pyunits.kW)
    m.fs.battery.four_hr_battery = Constraint(expr=m.fs.battery.nameplate_power * m.fs.battery.duration == m.fs.battery.nameplate_energy)
    return m.fs.battery


def add_h2_tank(m, tank_type="simple", valve_outlet_bar=None, length_m=None):
    """
    Adds a Hydrogen Tank unit to the flowsheet, with 3 options for the model type: `simple`, `detailed` and `detailed-valve`.

    For the `simple` type, the model is linear with no energy balances.
    For the `detailed` type, the model is nonlinear with energy balances deactivated. The capacity of the tank is determined by its 
    geometry.
    For the `detailed-valve` type, the `detailed` model is used, but with energy balances active and with a Valve unit. 
    The valve coefficient is pulled from load_parameters but needs to be be tuned per application.

    Args:
        m: existing ConcreteModel with a flowsheet `fs`
        tank_type: `simple`, `detailed`, or `detailed-valve`
        valve_outlet_bar: required if not using `simple` type, outlet pressure of the valve
        length_m: required if using `detailed` type, length of tank
    Returns:
        tank (and valve) unit model(s) in flowsheet
    """
    if "detailed" in tank_type:
        m.fs.h2_tank = HydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})
        m.fs.h2_tank.tank_diameter.fix(0.1)
        m.fs.h2_tank.tank_length.fix(length_m)
        m.fs.h2_tank.control_volume.properties_in[0].pressure.setub(max_pressure_bar * 1e5)
        m.fs.h2_tank.control_volume.properties_out[0].pressure.setub(max_pressure_bar * 1e5)
        m.fs.h2_tank.previous_state[0].pressure.setub(max_pressure_bar * 1e5)
        if tank_type == "detailed-valve":
            m.fs.tank_valve = Valve(
                default={
                    "valve_function_callback": ValveFunctionType.linear,
                    "property_package": m.fs.h2ideal_props,
                }
            )
            m.fs.tank_to_valve = Arc(
                source=m.fs.h2_tank.outlet,
                destination=m.fs.tank_valve.inlet
            )
            m.fs.tank_valve.outlet.pressure[0].fix(valve_outlet_bar * 1e5)
            # tune valve's coefficient of flow to match the condition
            m.fs.tank_valve.Cv.fix(valve_cv)
            # unfixing valve opening. This allows for controlling both pressure and flow at the outlet of the valve
            m.fs.tank_valve.valve_opening[0].unfix()
            m.fs.tank_valve.valve_opening[0].setlb(0)
        else:
            m.fs.h2_tank.energy_balances.deactivate()
    elif tank_type == "simple":
        m.fs.h2_tank = SimpleHydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})
        m.fs.h2_tank.outlet_to_turbine.mole_frac_comp[0, "hydrogen"].fix(1)
        m.fs.h2_tank.outlet_to_pipeline.mole_frac_comp[0, "hydrogen"].fix(1)
    else:
        raise ValueError(f"Unrecognized tank_type {tank_type}")

    m.fs.h2_tank.dt[0].fix(timestep_hrs * 3600)

    return m.fs.h2_tank


def add_h2_turbine(m, inlet_pres_bar):
    """
    Adds a hydrogen turbine unit to the flowsheet, using the h2_reaction_props property package.

    A translator and mixer unit is added also. The mixer adds an `air_feed` and an `purchased_hydrogen_feed` to the `hydrogen_feed`.
    The air_feed contains oxygen at a fixed ratio relative to the inlet hydrogen flow rate, `air_h2_ratio`, from load_parameters.
    .The purchased_hydrogen_feed allows the turbine hydrogen inlet flow rate to be `h2_turb_min_flow`, which helps
    with the model solve.

    The `compressor_dp` is fixed from load_parameters, the compressor's isentropic efficiency is 0.86, 
    the stoichiometric conversion rate of hydrogen is 0.99, and the turbine's isentropic efficiency is 0.89.

    The turbine's `electricity` is an Expression taking the difference between the turbine's and compressor's work.

    Args:
        m: existing ConcreteModel with a flowsheet `fs`
        inlet_pres_bar: operating pressure of inlet air and hydrogen to the mixer
    Returns:
        tank (and valve, if applicable) unit model(s) in flowsheet
    """
    m.fs.h2turbine_props = GenericParameterBlock(default=hturbine_config)

    m.fs.reaction_params = h2_reaction_props.H2ReactionParameterBlock(
        default={"property_package": m.fs.h2turbine_props})

    # Add translator block
    m.fs.translator = Translator(
        default={"inlet_property_package": m.fs.h2ideal_props,
                 "outlet_property_package": m.fs.h2turbine_props})

    m.fs.translator.eq_flow_hydrogen = Constraint(
        expr=m.fs.translator.inlet.flow_mol[0] ==
        m.fs.translator.outlet.flow_mol[0]
    )

    m.fs.translator.eq_temperature = Constraint(
        expr=m.fs.translator.inlet.temperature[0] ==
        m.fs.translator.outlet.temperature[0]
    )

    m.fs.translator.eq_pressure = Constraint(
        expr=m.fs.translator.inlet.pressure[0] ==
        m.fs.translator.outlet.pressure[0]
    )

    m.fs.translator.outlet.mole_frac_comp[0, "hydrogen"].fix(0.99)
    m.fs.translator.outlet.mole_frac_comp[0, "oxygen"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "argon"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "nitrogen"].fix(0.01/4)
    m.fs.translator.outlet.mole_frac_comp[0, "water"].fix(0.01/4)

    m.fs.translator.inlet.pressure[0].setub(max_pressure_bar * 1e5)
    m.fs.translator.outlet.pressure[0].setub(max_pressure_bar * 1e5)

    # Add mixer block
    # purchased_hydrogen_feed as slack for turbine inlet flow mol to be nonzero
    m.fs.mixer = Mixer(
        default={
    # using minimize pressure for all inlets and outlet of the mixer
    # because pressure of inlets is already fixed in flowsheet, using equality will over-constrain
            "momentum_mixing_type": MomentumMixingType.minimize,
            "property_package": m.fs.h2turbine_props,
            "inlet_list":
                ["air_feed", "hydrogen_feed", "purchased_hydrogen_feed"]}
    )

    m.fs.mixer.air_feed.temperature[0].fix(pem_temp)
    m.fs.mixer.air_feed.pressure[0].fix(inlet_pres_bar * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)
    m.fs.mixer.purchased_hydrogen_feed.pressure[0].fix(inlet_pres_bar * 1e5)
    m.fs.mixer.purchased_hydrogen_feed.temperature[0].fix(pem_temp)
    m.fs.mixer.purchased_hydrogen_feed.mole_frac_comp[0, "hydrogen"].fix(0.99)
    m.fs.mixer.purchased_hydrogen_feed.mole_frac_comp[0, "oxygen"].fix(0.01/4)
    m.fs.mixer.purchased_hydrogen_feed.mole_frac_comp[0, "argon"].fix(0.01/4)
    m.fs.mixer.purchased_hydrogen_feed.mole_frac_comp[0, "nitrogen"].fix(0.01/4)
    m.fs.mixer.purchased_hydrogen_feed.mole_frac_comp[0, "water"].fix(0.01/4)

    m.fs.mixer.mixed_state[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.mixer.air_feed_state[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.mixer.hydrogen_feed_state[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.mixer.purchased_hydrogen_feed_state[0].pressure.setub(max_pressure_bar * 1e5)

    m.fs.mixer.air_h2_ratio = Constraint(
        expr=m.fs.mixer.air_feed.flow_mol[0] == air_h2_ratio * (
                m.fs.mixer.purchased_hydrogen_feed.flow_mol[0] + m.fs.mixer.hydrogen_feed.flow_mol[0]))

    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].setlb(h2_turb_min_flow / 2)

    m.fs.translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )

    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})
    m.fs.h2_turbine.compressor.deltaP.fix(compressor_dp * 1e5)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

    # Specify the Stoichiometric Conversion Rate of hydrogen
    # in the equation shown below
    # H2(g) + O2(g) --> H2O(g) + energy
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    m.fs.h2_turbine.turbine.deltaP.fix(-compressor_dp * 1e5)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    # add operating constraints
    m.fs.h2_turbine.electricity = Expression(m.fs.config.time,
                                             rule=lambda b, t: (-b.turbine.work_mechanical[t] - b.compressor.work_mechanical[t]) * 1e-3)

    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    return m.fs.h2_turbine, m.fs.mixer, m.fs.translator


def create_model(wind_mw, pem_bar, batt_mw, tank_type, tank_length_m, turb_inlet_bar, wind_resource_config=None):
    """
    Creates a Flowsheet Pyomo model that puts together the Wind unit model with optional PEM, Hydrogen Tank, and Hydrogen Turbine unit models.

    The input parameters determine the size of the technologies by fixing the appropriate variable. If the size parameter is None, the technology
    will not be added.

    The wind is first split among its output destinations: grid, battery and PEM with an ElectricalSplitter unit model. 
    After the PEM, a tank and turbine may be added. The `simple` tank model includes outlet ports for hydrogen flows to the turbine and the pipeline.
    The `detailed` tank model uses a Splitter unit model to split the material and energy flows to the turbine and to hydrogen sales.

    Args:
        wind_mw: wind farm capacity
        pem_bar: operating pressure of the PEM  
        batt_mw: nameplate power
        tank_type: `simple`, `detailed`, or `detailed-valve`. See `add_h2_tank` for descriptions
        tank_length_m: required if using `detailed` tank type, length of tank
        turb_inlet_bar: operating inlet pressure of hydrogen to turbine
        wind_resource_config: dictionary of Windpower Config keys (`resource_speed`, `resource_probability_density`) and ConfigValues. See `add_wind` for description
    
    """
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    wind = add_wind(m, wind_mw, wind_resource_config)
    wind_output_dests = ["grid"]

    if pem_bar is not None:
        pem, _ = add_pem(m, pem_bar)
        wind_output_dests.append("pem")

    if batt_mw is not None:
        battery = add_battery(m, batt_mw)
        wind_output_dests.append("battery")

    if tank_length_m is not None:
        h2_tank = add_h2_tank(m, tank_type, pem_bar, tank_length_m)

    if turb_inlet_bar is not None and tank_length_m is not None:
        add_h2_turbine(m, turb_inlet_bar)

    # Set up where wind output flows to
    if len(wind_output_dests) > 1:
        m.fs.splitter = ElectricalSplitter(default={"outlet_list": wind_output_dests})
        m.fs.wind_to_splitter = Arc(source=wind.electricity_out, dest=m.fs.splitter.electricity_in)

    if "pem" in wind_output_dests:
        m.fs.splitter_to_pem = Arc(source=m.fs.splitter.pem_port, dest=pem.electricity_in)
    if "battery" in wind_output_dests:
        m.fs.splitter_to_battery = Arc(source=m.fs.splitter.battery_port, dest=battery.power_in)

    if hasattr(m.fs, "h2_tank"):
        m.fs.pem_to_tank = Arc(source=pem.outlet, dest=h2_tank.inlet)

    if hasattr(m.fs, "h2_turbine"):
        if tank_type == 'simple':
            m.fs.h2_tank_to_turb = Arc(source=m.fs.h2_tank.outlet_to_turbine,
                                       destination=m.fs.translator.inlet)
        else:
            m.fs.h2_splitter = Separator(default={"property_package": m.fs.h2ideal_props,
                                                "outlet_list": ["sold", "turbine"]})
            if tank_type == 'detailed-valve':
                m.fs.valve_to_h2_splitter = Arc(source=m.fs.tank_valve.outlet,
                                                destination=m.fs.h2_splitter.inlet)
            else:
                m.fs.valve_to_h2_splitter = Arc(source=m.fs.h2_tank.outlet,
                                                destination=m.fs.h2_splitter.inlet)
            # Set up where hydrogen from tank flows to
            m.fs.h2_splitter_to_turb = Arc(source=m.fs.h2_splitter.turbine,
                                        destination=m.fs.translator.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # Scaling factors, set mostly to 1 for now
    elec_sf = 1
    iscale.set_scaling_factor(m.fs.windpower.electricity, elec_sf)
    if hasattr(m.fs, "splitter"):
        iscale.set_scaling_factor(m.fs.splitter.electricity, elec_sf)
        iscale.set_scaling_factor(m.fs.splitter.grid_elec, elec_sf)

    if hasattr(m.fs, "battery"):
        iscale.set_scaling_factor(m.fs.splitter.battery_elec, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.elec_in, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.elec_out, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.nameplate_power, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.nameplate_energy, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.initial_state_of_charge, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.initial_energy_throughput, elec_sf)
        iscale.set_scaling_factor(m.fs.battery.state_of_charge, elec_sf)

    if hasattr(m.fs, "pem"):
        iscale.set_scaling_factor(m.fs.splitter.pem_elec, elec_sf)
        iscale.set_scaling_factor(m.fs.pem.electricity, elec_sf)

    if hasattr(m.fs, "h2_turbine"):
        iscale.set_scaling_factor(m.fs.mixer.minimum_pressure, 1)
        iscale.set_scaling_factor(m.fs.mixer.air_feed_state[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.mixer.hydrogen_feed_state[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.mixer.purchased_hydrogen_feed_state[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.mixer.mixed_state[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.mixer.hydrogen_feed_state[0.0].flow_mol_phase['Vap'], 1)

        iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.work, 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.compressor.properties_isentropic[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.rate_reaction_extent[0, 'R1'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.work, 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.properties_isentropic[0.0].enth_mol_phase['Vap'], 1)

    iscale.calculate_scaling_factors(m)
    return m
