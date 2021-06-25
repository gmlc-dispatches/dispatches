#############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform to Advance Tightly
# Coupled Hybrid Energy Systems program (DISPATCHES), and is copyright © 2021 by the software owners:
# The Regents of the University of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable Energy, LLC, Battelle
# Energy Alliance, LLC, University of Notre Dame du Lac, et al. All rights reserved.

# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the
# U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted
# for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license
# in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit other to do so.
##############################################################################
"""
Renewable Energy Flowsheet
Author: Darice Guittet
Date: June 7, 2021
"""

import matplotlib.pyplot as plt
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TransformationFactory,
                           value)
from pyomo.network import Arc
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.generic_models.unit_models import Translator, Mixer

from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration as hturbine_config
import dispatches.models.nuclear_case.properties.h2_reaction \
    as h2_reaction_props

from dispatches.models.nuclear_case.unit_models.hydrogen_turbine_unit import HydrogenTurbine
from dispatches.models.nuclear_case.unit_models.hydrogen_tank import HydrogenTank
from dispatches.models.renewables_case.pem_electrolyzer import PEM_Electrolyzer
from dispatches.models.renewables_case.elec_splitter import ElectricalSplitter
from dispatches.models.renewables_case.battery import BatteryStorage
from dispatches.models.renewables_case.wind_power import Wind_Power

timestep_hrs = 1
H2_mass = 2.016 / 1000


def add_wind(m):
    resource_timeseries = dict()
    for time in list(m.fs.config.time.data()):
        # ((wind m/s, wind degrees from north clockwise, probability), )
        resource_timeseries[time] = ((10, 180, 0.5),
                                     (24, 180, 0.5))
    wind_config = {'resource_probability_density': resource_timeseries}

    m.fs.windpower = Wind_Power(default=wind_config)
    m.fs.windpower.system_capacity.fix(20000)   # kW
    return m.fs.windpower


def add_pem(m):
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    m.fs.pem = PEM_Electrolyzer(
        default={"property_package": m.fs.h2ideal_props})

    # Conversion of kW to mol/sec of H2. (elec*elec_to_mol) based on H-tec design of 54.517kW-hr/kg
    m.fs.pem.electricity_to_mol.fix(0.002527406)
    return m.fs.pem, m.fs.h2ideal_props


def add_battery(m):
    m.fs.battery = BatteryStorage()
    m.fs.battery.dt.set_value(timestep_hrs)
    m.fs.battery.nameplate_power.set_value(1000)
    m.fs.battery.nameplate_energy.fix(4000)       # kW
    return m.fs.battery


def add_h2_tank(m):
    m.fs.h2_tank = HydrogenTank(default={"property_package": m.fs.h2ideal_props, "dynamic": False})

    m.fs.h2_tank.tank_diameter.fix(0.1)
    m.fs.h2_tank.tank_length.fix(0.3)

    m.fs.h2_tank.dt[0].fix(timestep_hrs * 3600)

    m.fs.h2_tank.inlet.pressure.setub(1e15)
    m.fs.h2_tank.outlet.pressure.setub(1e15)
    return m.fs.h2_tank


def add_h2_turbine(m):
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

    # Add mixer block
    m.fs.mixer = Mixer(
        default={"property_package": m.fs.h2turbine_props,
                 "inlet_list":
                 ["air_feed", "hydrogen_feed"]}
    )
    m.fs.mixer.air_feed.temperature[0].fix(300)
    m.fs.mixer.air_feed.pressure[0].fix(1e5)
    m.fs.mixer.air_feed.mole_frac_comp[0, "oxygen"].fix(0.2054)
    m.fs.mixer.air_feed.mole_frac_comp[0, "argon"].fix(0.0032)
    m.fs.mixer.air_feed.mole_frac_comp[0, "nitrogen"].fix(0.7672)
    m.fs.mixer.air_feed.mole_frac_comp[0, "water"].fix(0.0240)
    m.fs.mixer.air_feed.mole_frac_comp[0, "hydrogen"].fix(2e-4)

    # Add the hydrogen turbine
    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    m.fs.h2_turbine.compressor.deltaP.fix(2.401e6)
    m.fs.h2_turbine.compressor.efficiency_isentropic.fix(0.86)

    # Specify the Stoichiometric Conversion Rate of hydrogen
    # in the equation shown below
    # H2(g) + O2(g) --> H2O(g) + energy
    # Complete Combustion
    m.fs.h2_turbine.stoic_reactor.conversion.fix(0.99)

    m.fs.h2_turbine.turbine.deltaP.fix(-2.401e6)
    m.fs.h2_turbine.turbine.efficiency_isentropic.fix(0.89)

    m.fs.H2_production = Expression(
        expr=m.fs.pem.outlet.flow_mol[0] * H2_mass)

    # add arcs
    m.fs.translator_to_mixer = Arc(
        source=m.fs.translator.outlet,
        destination=m.fs.mixer.hydrogen_feed
    )

    m.fs.mixer_to_turbine = Arc(
        source=m.fs.mixer.outlet,
        destination=m.fs.h2_turbine.compressor.inlet
    )

    return m.fs.h2_turbine, m.fs.mixer, m.fs.translator


def create_model():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    wind = add_wind(m)

    pem, pem_properties = add_pem(m)

    battery = add_battery(m)

    h2_tank = add_h2_tank(m)

    h2_turbine, h2_mixer, h2_turbine_translator = add_h2_turbine(m)

    m.fs.splitter = ElectricalSplitter(default={"outlet_list": ["pem", "battery"]})

    # Set up network
    m.fs.wind_to_splitter = Arc(source=wind.electricity_out, dest=m.fs.splitter.electricity_in)
    m.fs.splitter_to_pem = Arc(source=m.fs.splitter.pem_port, dest=pem.electricity_in)
    m.fs.splitter_to_battery = Arc(source=m.fs.splitter.battery_port, dest=battery.power_in)

    m.fs.pem_to_tank = Arc(source=pem.outlet, dest=h2_tank.inlet)
    m.fs.h2_tank_to_translator = Arc(source=m.fs.h2_tank.outlet, destination=m.fs.translator.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def set_initial_conditions(m):
    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)

    # Fix the outlet flow to zero for tank filling type operation
    m.fs.h2_tank.previous_state[0].temperature.fix(300)
    m.fs.h2_tank.previous_state[0].pressure.fix(1e5)

    return m


def initialize_model(m):
    m.fs.pem.initialize()
    m.fs.h2_tank.initialize()

    propagate_state(m.fs.h2_tank_to_translator)
    m.fs.translator.initialize()

    propagate_state(m.fs.translator_to_mixer)
    m.fs.mixer.initialize()

    propagate_state(m.fs.mixer_to_turbine)
    m.fs.h2_turbine.initialize()
    return m


battery_discharge_kw = [0, 0, 0]
h2_out_mol_per_s = [0.1, 0.1, 0.2]


def update_control_vars(m, i):
    batt_kw = battery_discharge_kw[i]
    if batt_kw > 0:
        m.fs.battery.elec_in.fix(0)
        m.fs.battery.elec_out.fix(batt_kw)
    else:
        m.fs.battery.elec_in.fix(-batt_kw)
        m.fs.battery.elec_out.fix(0)

    h2_flow = h2_out_mol_per_s[i]
    # m.fs.h2_tank.outlet.flow_mol.fix(h2_flow)

    m.fs.mixer.air_feed.flow_mol[0].fix(2650)


def update_state(m):
    m.fs.battery.initial_state_of_charge.fix(value(m.fs.battery.state_of_charge[0]))
    m.fs.battery.initial_energy_throughput.fix(value(m.fs.battery.energy_throughput[0]))

    m.fs.h2_tank.previous_state[0].pressure.fix(value(m.fs.h2_tank.control_volume.properties_out[0].pressure))
    m.fs.h2_tank.previous_state[0].temperature.fix(value(m.fs.h2_tank.control_volume.properties_out[0].temperature))


if __name__ == "__main__":

    m = create_model()
    wind_out_kw = []
    batt_in_kw = []
    batt_soc = []
    pem_in_kw = []
    tank_in_mol_per_s = []
    tank_holdup_mol = []
    tank_out_mol_per_s = []
    m = set_initial_conditions(m)
    for i in range(0, 3):
        update_control_vars(m, i)
        m = initialize_model(m)

        assert_units_consistent(m)
        print(degrees_of_freedom(m))

        solver = SolverFactory('ipopt')
        res = solver.solve(m, tee=False)

        wind_out_kw.append(value(m.fs.windpower.electricity[0]))

        batt_in_kw.append(value(m.fs.battery.elec_in[0]))
        batt_soc.append(value(m.fs.battery.state_of_charge[0]))
        # print(value(m.fs.battery.elec_out[0]))

        pem_in_kw.append(value(m.fs.splitter.pem_elec[0]))
        print("inlet flow mol", value(m.fs.h2_tank.inlet.flow_mol[0]))
        print("outlet flow mol", value(m.fs.h2_tank.outlet.flow_mol[0]))

        print("previous_material_holdup", value(m.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')]))
        print("material_holdup", value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]))
        print('material_accumulation', value(m.fs.h2_tank.material_accumulation[0, ('Vap', 'hydrogen')]))

        print('previous_energy_holdup', value(m.fs.h2_tank.previous_energy_holdup[0, ('Vap')]))
        print('energy_holdup', value(m.fs.h2_tank.energy_holdup[0, ('Vap')]))
        print('energy_accumulation', value(m.fs.h2_tank.energy_accumulation[0, ('Vap')]))
        print('properties out pres', value(m.fs.h2_tank.control_volume.properties_out[0].pressure))
        print('properties out temp', value(m.fs.h2_tank.control_volume.properties_out[0].temperature))

        tank_in_mol_per_s.append(value(m.fs.h2_tank.inlet.flow_mol[0]))
        tank_out_mol_per_s.append(value(m.fs.h2_tank.outlet.flow_mol[0]))
        tank_holdup_mol.append(value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]))

        print("#### Tank ###")
        m.fs.h2_tank.report()

        print("#### Mixer ###")
        m.fs.mixer.report()

        print("#### Hydrogen Turbine ###")
        m.fs.h2_turbine.compressor.report()
        m.fs.h2_turbine.stoic_reactor.report()
        m.fs.h2_turbine.turbine.report()

        update_state(m)

    print(tank_in_mol_per_s)

    n = len(battery_discharge_kw) - 1

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

