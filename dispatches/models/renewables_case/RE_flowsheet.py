##############################################################################
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
#
##############################################################################
"""
Renewable Energy Flowsheet
Author: Darice Guittet
Date: June 7, 2021
"""

from idaes.core.util.scaling import badly_scaled_var_generator
import matplotlib.pyplot as plt
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
from pyomo.util.check_units import assert_units_consistent
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.contrib.parmest.ipopt_solver_wrapper import ipopt_solve_with_stats
import idaes.core.util.scaling as iscale

import idaes.logger as idaeslog
from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.generic_models.unit_models import (Translator,
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

from idaes.generic_models.unit_models.product import Product
from idaes.generic_models.unit_models.separator import Separator
from dispatches.models.nuclear_case.unit_models.hydrogen_turbine_unit import HydrogenTurbine
from dispatches.models.nuclear_case.unit_models.hydrogen_tank import HydrogenTank
from dispatches.models.renewables_case.load_parameters import *
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter
from dispatches.models.renewables_case.battery import BatteryStorage
from dispatches.models.renewables_case.wind_power import Wind_Power

timestep_hrs = 1
H2_mass = 2.016 / 1000

PEM_temp = 300
H2_turb_pressure_bar = 24.7
max_pressure_bar = 700


def add_wind(m, wind_mw, wind_config=None):
    resource_timeseries = dict()
    for time in list(m.fs.config.time.data()):
        # ((wind m/s, wind degrees from north clockwise, probability), )
        resource_timeseries[time] = ((10, 180, 0.5),
                                     (24, 180, 0.5))
    if wind_config is None:
        wind_config = {'resource_probability_density': resource_timeseries}

    m.fs.windpower = Wind_Power(default=wind_config)
    m.fs.windpower.system_capacity.fix(wind_mw * 1e3)   # kW
    return m.fs.windpower


def add_pem(m, outlet_pressure_bar):
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
    m.fs.pem.outlet.temperature.fix(PEM_temp)
    return m.fs.pem, m.fs.h2ideal_props


def add_battery(m, batt_mw):
    m.fs.battery = BatteryStorage()
    m.fs.battery.charging_eta.set_value(1)
    m.fs.battery.discharging_eta.set_value(1)
    m.fs.battery.dt.set_value(timestep_hrs)
    m.fs.battery.nameplate_power.fix(batt_mw * 1e3)
    m.fs.battery.duration = Param(default=4, mutable=True, units=pyunits.kWh/pyunits.kW)
    m.fs.battery.four_hr_battery = Constraint(expr=m.fs.battery.nameplate_power * m.fs.battery.duration == m.fs.battery.nameplate_energy)
    return m.fs.battery


def add_h2_tank(m, pem_pres_bar, length_m, valve_Cv):
    m.fs.h2_tank = HydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})
    m.fs.h2_tank.tank_diameter.fix(0.1)
    m.fs.h2_tank.tank_length.fix(length_m)
    m.fs.h2_tank.control_volume.properties_in[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.h2_tank.control_volume.properties_out[0].pressure.setub(max_pressure_bar * 1e5)
    m.fs.h2_tank.previous_state[0].pressure.setub(max_pressure_bar * 1e5)
    if not use_simple_h2_tank:
        # hydrogen tank valve
        m.fs.tank_valve = Valve(
            default={
                "valve_function_callback": ValveFunctionType.linear,
                "property_package": m.fs.h2ideal_props,
            }
        )
        # connect tank to the valve
        m.fs.tank_to_valve = Arc(
            source=m.fs.h2_tank.outlet,
            destination=m.fs.tank_valve.inlet
        )
        m.fs.tank_valve.outlet.pressure[0].fix(pem_pres_bar * 1e5)
        # NS: tuning valve's coefficient of flow to match the condition
        m.fs.tank_valve.Cv.fix(valve_Cv)
        # NS: unfixing valve opening. This allows for controlling both pressure
        # and flow at the outlet of the valve
        m.fs.tank_valve.valve_opening[0].unfix()
        m.fs.tank_valve.valve_opening[0].setlb(0)
    # else:
    #     m.fs.h2_tank = SimpleHydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})
    #     m.fs.h2_tank.outlet_to_turbine.mole_frac_comp[0, "hydrogen"].fix(1)
    #     m.fs.h2_tank.outlet_to_pipeline.mole_frac_comp[0, "hydrogen"].fix(1)

    m.fs.h2_tank.dt[0].fix(timestep_hrs * 3600)

    return m.fs.h2_tank


def add_h2_turbine(m, pem_pres_bar):
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

    m.fs.translator.mole_frac_hydrogen = Constraint(
        expr=m.fs.translator.outlet.mole_frac_comp[0, "hydrogen"] == 0.99
    )
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

    m.fs.mixer.air_feed.temperature[0].fix(PEM_temp)
    m.fs.mixer.air_feed.pressure[0].fix(pem_pres_bar * 1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)
    m.fs.mixer.purchased_hydrogen_feed.pressure[0].fix(pem_pres_bar * 1e5)
    m.fs.mixer.purchased_hydrogen_feed.temperature[0].fix(PEM_temp)
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
                m.fs.mixer.purchased_hydrogen_feed.flow_mol[0] +
                                                                m.fs.mixer.hydrogen_feed.flow_mol[0]))
    #m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].setlb(h2_turb_min_flow - 1e-4)

    # add arcs
    m.fs.translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )
    # Return early without adding Turbine
    # return None, m.fs.mixer, m.fs.translator

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})
    min_mole_frac = 1e-3
    m.fs.h2_turbine.compressor.deltaP.fix(compressor_dp * 1e5)
    # m.fs.h2_turbine.compressor.ratioP.fix(30)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.compressor.properties_isentropic[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)


    # Specify the Stoichiometric Conversion Rate of hydrogen
    # in the equation shown below
    # H2(g) + O2(g) --> H2O(g) + energy

    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    # m.fs.h2_turbine.turbine.deltaP.fix(-(H2_turb_pressure_bar - .101325) * 1e5)
    # m.fs.h2_turbine.turbine.deltaP.setub(-(H2_turb_pressure_bar - .101325) * 1e5 * 0.75)
    # m.fs.h2_turbine.turbine.deltaP.setlb(-(H2_turb_pressure_bar - .101325) * 1e5 * 1.25)

    min_mole_frac = 1e-5
    m.fs.h2_turbine.turbine.deltaP.fix(-compressor_dp * 1e5)

    # m.fs.h2_turbine.turbine.ratioP.fix(1/30)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.control_volume.properties_in[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'hydrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'argon'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'nitrogen'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'water'].setlb(min_mole_frac)
    # m.fs.h2_turbine.turbine.properties_isentropic[0].mole_frac_phase_comp['Vap', 'oxygen'].setlb(min_mole_frac)

    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    return m.fs.h2_turbine, m.fs.mixer, m.fs.translator


def create_model(wind_mw, pem_bar, batt_mw, valve_cv, tank_len_m, h2_turb_bar, wind_resource_config=None, verbose=False):
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    wind = add_wind(m, wind_mw, wind_resource_config)
    wind_output_dests = ["grid"]

    if pem_bar is not None:
        pem, pem_properties = add_pem(m, pem_bar)
        wind_output_dests.append("pem")

    if batt_mw is not None:
        battery = add_battery(m, batt_mw)
        wind_output_dests.append("battery")

    if valve_cv is not None and tank_len_m is not None:
        h2_tank = add_h2_tank(m, pem_bar, tank_len_m, valve_cv)

    if h2_turb_bar is not None and tank_len_m is not None:
        h2_turbine, h2_mixer, h2_turbine_translator = add_h2_turbine(m, pem_bar)

    # Set up where wind output flows to
    m.fs.wind_to_grid = Var(m.fs.config.time, within=NonNegativeReals, initialize=0.0, units=pyunits.kW)
    if len(wind_output_dests) > 1:
        m.fs.wind_to_grid_port = Port(noruleinit=True, doc="Electricity flow from wind to grid")
        m.fs.wind_to_grid_port.add(m.fs.wind_to_grid, "electricity")
        m.fs.splitter = ElectricalSplitter(default={"outlet_list": wind_output_dests})
        m.fs.wind_to_splitter = Arc(source=wind.electricity_out, dest=m.fs.splitter.electricity_in)
        m.fs.splitter_to_grid = Arc(source=m.fs.splitter.grid_port, dest=m.fs.wind_to_grid_port)

    if "pem" in wind_output_dests:
        m.fs.splitter_to_pem = Arc(source=m.fs.splitter.pem_port, dest=pem.electricity_in)
    if "battery" in wind_output_dests:
        m.fs.splitter_to_battery = Arc(source=m.fs.splitter.battery_port, dest=battery.power_in)

    if hasattr(m.fs, "h2_tank"):
        m.fs.pem_to_tank = Arc(source=pem.outlet, dest=h2_tank.inlet)

    if hasattr(m.fs, "h2_turbine"):
        m.fs.h2_splitter = Separator(default={"property_package": m.fs.h2ideal_props,
                                              "outlet_list": ["sold", "turbine"]})
        if not use_simple_h2_tank:
            m.fs.valve_to_h2_splitter = Arc(source=m.fs.tank_valve.outlet,
                                            destination=m.fs.h2_splitter.inlet)
        else:
            m.fs.valve_to_h2_splitter = Arc(source=m.fs.h2_tank.outlet,
                                            destination=m.fs.h2_splitter.inlet)
        # Set up where hydrogen from tank flows to
        m.fs.h2_splitter_to_turb = Arc(source=m.fs.h2_splitter.turbine,
                                       destination=m.fs.translator.inlet)

        m.fs.tank_sold = Product(default={"property_package": m.fs.h2ideal_props})

        m.fs.h2_splitter_to_sold = Arc(source=m.fs.h2_splitter.sold,
                                       destination=m.fs.tank_sold.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)

    # Scaling factors, set mostly to 1 for now
    iscale.set_scaling_factor(m.fs.windpower.electricity, 1e-5)
    iscale.set_scaling_factor(m.fs.wind_to_grid, 1e-5)
    if hasattr(m.fs, "splitter"):
        iscale.set_scaling_factor(m.fs.splitter.electricity, 1e-5)
        iscale.set_scaling_factor(m.fs.splitter.grid_elec, 1e-5)

    if hasattr(m.fs, "battery"):
        iscale.set_scaling_factor(m.fs.splitter.battery_elec, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.elec_in, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.elec_out, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.nameplate_power, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.nameplate_energy, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.initial_state_of_charge, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.initial_energy_throughput, 1e-5)
        iscale.set_scaling_factor(m.fs.battery.state_of_charge, 1e-5)

    if hasattr(m.fs, "pem"):
        iscale.set_scaling_factor(m.fs.splitter.pem_elec, 1e-5)
        iscale.set_scaling_factor(m.fs.pem.electricity, 1e-5)

    if hasattr(m.fs, "h2_tank"):
        iscale.set_scaling_factor(m.fs.h2_tank.material_holdup[0.0, 'Vap', 'hydrogen'], 1)
        iscale.set_scaling_factor(m.fs.h2_tank.energy_holdup[0.0, 'Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_tank.previous_material_holdup[0.0, 'Vap', 'hydrogen'], 1)
        iscale.set_scaling_factor(m.fs.h2_tank.previous_energy_holdup[0.0, 'Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_tank.control_volume.deltaP, 1)
        iscale.set_scaling_factor(m.fs.h2_tank.control_volume.volume, 1)
        iscale.set_scaling_factor(m.fs.h2_tank.tank_length, 1e-3)
        iscale.set_scaling_factor(m.fs.h2_tank.tank_diameter, 1e-3)

    if hasattr(m.fs, "tank_valve"):
        iscale.set_scaling_factor(m.fs.tank_valve.valve_opening, 1e-3)
        iscale.set_scaling_factor(m.fs.h2_tank.control_volume.volume, 1)
        iscale.set_scaling_factor(m.fs.tank_valve.control_volume.work, 1)

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
        # iscale.set_scaling_factor(m.fs.h2_turbine.compressor.work_isentropic, 1e-6)
        # iscale.set_scaling_factor(m.fs.h2_turbine.compressor.properties_isentropic[0.0].mole_frac_phase_comp['Vap', 'hydrogen'], 1e3)
        # iscale.set_scaling_factor(m.fs.h2_turbine.compressor.control_volume.properties_in[0.0].mole_frac_phase_comp['Vap', 'hydrogen'], 1e3)
        iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.stoic_reactor.control_volume.rate_reaction_extent[0, 'R1'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_in[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_out[0.0].enth_mol_phase['Vap'], 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.work, 1)
        iscale.set_scaling_factor(m.fs.h2_turbine.turbine.properties_isentropic[0.0].enth_mol_phase['Vap'], 1)
        # iscale.set_scaling_factor(m.fs.h2_turbine.turbine.work_isentropic, 1e-6)
        # iscale.set_scaling_factor(m.fs.h2_turbine.turbine.properties_isentropic[0.0].mole_frac_phase_comp['Vap', 'hydrogen'], 1e3)
        # iscale.set_scaling_factor(m.fs.h2_turbine.turbine.control_volume.properties_in[0.0].mole_frac_phase_comp['Vap', 'hydrogen'], 1e3)

    iscale.calculate_scaling_factors(m)
    return m


def report_model(m):
    print("wind out kW", value(m.fs.windpower.electricity[0]))
    print("#### Splitter ###")
    m.fs.splitter.report()
    print("#### PEM ###")
    m.fs.pem.report()
    print("#### Tank ###")
    if hasattr(m.fs, "tank_valve"):
        m.fs.h2_tank.report()
        m.fs.tank_valve.report()
    if hasattr(m.fs, "mixer"):
        print("#### Mixer ###")
        m.fs.translator.report()
        m.fs.mixer.report()
    if hasattr(m.fs, "h2_turbine"):
        print("#### Hydrogen Turbine ###")
        m.fs.h2_turbine.report()


def plot_model(m):
    wind_out_kw = []
    batt_in_kw = []
    batt_soc = []
    pem_in_kw = []
    tank_in_mol_per_s = []
    tank_holdup_mol = []
    tank_out_mol_per_s = []
    turbine_work_net = []

    for i in list(m.fs.config.time.data()):
        wind_out_kw.append(value(m.fs.windpower.electricity[0]))
        batt_in_kw.append(value(m.fs.battery.elec_in[0]))
        batt_soc.append(value(m.fs.battery.state_of_charge[0]))
        pem_in_kw.append(value(m.fs.splitter.pem_elec[0]))
        tank_in_mol_per_s.append(value(m.fs.h2_tank.inlet.flow_mol[0]))
        tank_out_mol_per_s.append(value(m.fs.h2_tank.outlet.flow_mol[0]))
        tank_holdup_mol.append(value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]))
        turbine_work_net.append(value(m.fs.h2_turbine.turbine.work_mechanical[0]
                                      + m.fs.h2_turbine.compressor.work_mechanical[0]))

    n = len(list(m.fs.config.time.data())) - 1
    fig, ax = plt.subplots(3, 1)

    ax[0].set_title("Fixed & Control Vars")
    ax[0].plot(wind_out_kw, 'k', label="wind output [kW]")
    ax[0].plot(batt_in_kw, label="batt dispatch [kW]")
    ax[0].set_ylabel("Power [kW]")
    ax[0].grid()
    ax[0].legend(loc="upper left")
    ax01 = ax[0].twinx()
    ax01.plot(tank_out_mol_per_s, 'g', label="tank out H2 [mol/s]")
    ax01.set_ylabel("Flow [mol/s]")
    ax01.legend(loc='lower right')
    ax[0].set_xlim((0, n))
    ax01.set_xlim((0, n))

    ax[1].set_title("Electricity")
    ax[1].plot(pem_in_kw, 'orange', label="pem in [kW]")
    ax[1].set_ylabel("Power [kW]")
    ax[1].grid()
    ax[1].legend(loc="upper left")
    ax11 = ax[1].twinx()
    ax11.plot(batt_soc, 'purple', label="batt SOC [1]")
    ax11.set_ylabel("[1]")
    ax11.legend(loc='lower right')
    ax[1].set_xlim((0, n))
    ax11.set_xlim((0, n))

    ax[2].set_title("H2")
    ax[2].plot(tank_in_mol_per_s, 'g', label="pem into tank H2 [mol/s]")
    ax[2].set_ylabel("H2 Flow [mol/s]")
    ax[2].grid()
    ax[2].legend(loc="upper left")
    ax21 = ax[2].twinx()
    ax21.plot(tank_holdup_mol, 'r', label="tank holdup [mol]")
    ax21.set_ylabel("H2 Mols [mol]")
    ax21.legend(loc='lower right')
    ax[2].set_xlim((0, n))
    ax21.set_xlim((0, n))

    plt.xlabel("Hr")
    fig.tight_layout()
    plt.show()

