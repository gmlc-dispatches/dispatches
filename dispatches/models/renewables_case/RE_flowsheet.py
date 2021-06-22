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

from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TransformationFactory,
                           value)
from pyomo.network import Arc
from pyomo.environ import units

from idaes.core import FlowsheetBlock
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

    m.fs.translator = Translator(
        default={"inlet_property_package": m.fs.h2ideal_props,
                 "outlet_property_package": m.fs.h2turbine_props})

    m.fs.mixer = Mixer(
        default={"property_package": m.fs.h2turbine_props,
                 "inlet_list":
                     ["air_feed", "hydrogen_feed"]}
    )

    m.fs.h2_turbine = HydrogenTurbine(
        default={"property_package": m.fs.h2turbine_props,
                 "reaction_package": m.fs.reaction_params})

    m.fs.H2_mass = 2.016 / 1000

    m.fs.H2_production = Expression(
        expr=m.fs.pem.outlet.flow_mol[0] * m.fs.H2_mass)

    # Set hydrogen flow and mole frac
    m.fs.h2_turbine_translator.eq_flow_hydrogen = Constraint(
        expr=m.fs.h2_turbine_translator.inlet.flow_mol[0] ==
             m.fs.h2_turbine_translator.outlet.flow_mol[0]
    )

    m.fs.h2_turbine_translator.mole_frac_hydrogen = Constraint(
        expr=m.fs.h2_turbine_translator.outlet.mole_frac_comp[0, "hydrogen"] == 0.99
    )

    m.fs.h2_turbine_translator.eq_temperature = Constraint(
        expr=m.fs.h2_turbine_translator.inlet.temperature[0] ==
             m.fs.h2_turbine_translator.outlet.temperature[0]
    )

    m.fs.h2_turbine_translator.eq_pressure = Constraint(
        expr=m.fs.h2_turbine_translator.inlet.pressure[0] ==
             m.fs.h2_turbine_translator.outlet.pressure[0]
    )

    m.fs.h2_turbine_translator.outlet.mole_frac_comp[0, "oxygen"].fix(0.01/4)
    m.fs.h2_turbine_translator.outlet.mole_frac_comp[0, "argon"].fix(0.01/4)
    m.fs.h2_turbine_translator.outlet.mole_frac_comp[0, "nitrogen"].fix(0.01/4)
    m.fs.h2_turbine_translator.outlet.mole_frac_comp[0, "water"].fix(0.01/4)
    return m.fs.h2_turbine, m.fs.mixer, m.fs.h2_turbine_translator


def create_model():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    wind = add_wind(m)

    pem, pem_properties = add_pem(m)

    battery = add_battery(m)

    h2_tank = add_h2_tank(m)

    # h2_turbine, h2_mixer, h2_turbine_translator = add_h2_turbine(m)

    m.fs.splitter = ElectricalSplitter(default={"outlet_list": ["pem", "battery"]})

    # Set up network
    m.fs.wind_to_splitter = Arc(source=wind.electricity_out, dest=m.fs.splitter.electricity_in)
    m.fs.splitter_to_pem = Arc(source=m.fs.splitter.pem_port, dest=pem.electricity_in)
    m.fs.splitter_to_battery = Arc(source=m.fs.splitter.battery_port, dest=battery.power_in)

    m.fs.pem_to_tank = Arc(source=pem.outlet, dest=h2_tank.inlet)

    # m.fs.tank_to_h2_turbine_translator = Arc(source=h2_tank.outlet, dest=h2_turbine_translator.inlet)
    # m.fs.h2_turbine_translator_to_mixer = Arc(source=h2_turbine_translator.outlet, dest=h2_mixer.hydrogen_feed)
    # m.fs.mixer_to_turbine = Arc(source=h2_mixer.outlet, dest=h2_turbine.compressor.inlet)

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def set_initial_conditions(m):
    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    m.fs.battery.elec_in.fix(0)
    m.fs.battery.elec_out.fix(0)

    # Fix the outlet flow to zero for tank filling type operation
    m.fs.h2_tank.previous_state[0].temperature.fix(300)
    m.fs.h2_tank.previous_state[0].pressure.fix(1e5)
    m.fs.h2_tank.outlet.flow_mol.fix(0)

    return m


def initialize_model(m):
    m.fs.pem.initialize()
    m.fs.h2_tank.initialize()
    return m


from pyomo.util.check_units import assert_units_consistent
from idaes.core.util.model_statistics import degrees_of_freedom
import matplotlib.pyplot as plt

if __name__ == "__main__":

    m = create_model()
    wind_out_kw = []
    batt_in_kw = []
    pem_in_kw = []
    tank_in_mol = []
    tank_h2_mol = []
    for i in range(0, 3):
        m = set_initial_conditions(m)
        m = initialize_model(m)

        assert_units_consistent(m)
        print(degrees_of_freedom(m))

        solver = SolverFactory('ipopt')
        res = solver.solve(m, tee=False)

        wind_out_kw.append(value(m.fs.windpower.electricity[0]))

        batt_in_kw.append(value(m.fs.battery.elec_in[0]))
        # print(value(m.fs.battery.elec_out[0]))

        pem_in_kw.append(value(m.fs.splitter.pem_elec[0]))
        print("inlet flow mol", value(m.fs.pem.outlet.flow_mol[0]))

        print("previous_material_holdup", value(m.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')]))
        print("material_holdup", value(m.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]))
        print('material_accumulation', value(m.fs.h2_tank.material_accumulation[0, ('Vap', 'hydrogen')]))

        print('previous_energy_holdup', value(m.fs.h2_tank.previous_energy_holdup[0, ('Vap')]))
        print('energy_holdup', value(m.fs.h2_tank.energy_holdup[0, ('Vap')]))
        print('energy_accumulation', value(m.fs.h2_tank.energy_accumulation[0, ('Vap')]))
        print('properties out pres', value(m.fs.h2_tank.control_volume.properties_out[0].pressure))
        print('properties out temp', value(m.fs.h2_tank.control_volume.properties_out[0].temperature))

        tank_in_mol.append(value(m.fs.h2_tank.inlet.flow_mol[0]))

    print(tank_in_mol)

    plt.plot(wind_out_kw, label="wind out")
    plt.plot(batt_in_kw, label="batt in")
    plt.plot(pem_in_kw, label="pem in")
    plt.plot(tank_in_mol, label="tank in")

    plt.legend()
    plt.xlabel("Power [kW]")
    plt.ylabel("Hr")
    plt.show()

