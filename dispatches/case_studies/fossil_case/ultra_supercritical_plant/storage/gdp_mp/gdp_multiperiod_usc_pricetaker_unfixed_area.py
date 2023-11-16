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

"""This script uses the IDAES multiperiod class to create a steady
state GDP multiperiod model for the integrated ultra-supercritical
power plant GDP model. The purpose of this script is to create a GDP
multiperiod model that can be use for market analysis using a
pricetaker assumption. The integrated storage with ultra-supercritical
power plant model is used a steady state model for creating the
multiperiod model.

"""

import json

import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.environ import (Constraint, NonNegativeReals, Var)

from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale
from idaes.core.solvers.get_solver import get_solver


__author__ = "Soraya Rawlings"

# Use GDP design for charge and discharge heat exchanger from 4-12
# disjunctions model when True. If False, use the GDP design from 4-5
# disjunctions model.
new_design = True

if new_design:
    print('>>>>> Solving for new storage design')
    import usc_storage_gdp_mp_unfixed_area_new_storage_design as usc_gdp
    # Add design data from .json file
    data_path = 'uscp_design_data_new_storage_design.json'
else:
    print('>>>>> Solving for original storage design')
    import usc_storage_gdp_mp_unfixed_area as usc_gdp
    # Add design data from .json file
    data_path = 'uscp_design_data.json'


with open(data_path) as design_data:
    design_data_dict = json.load(design_data)

min_power = design_data_dict["plant_min_power"] # in MW
max_power = design_data_dict["plant_max_power"] # in MW


def create_ss_model():

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Add options to model
    deact_arcs_after_init = True # needed for GDP model
    method = "with_efficiency" # adds boiler and cycle efficiencies
    load_init_file = False

    # Add data from .json data file
    cold_salt_temp = design_data_dict["cold_salt_temperature"] # in K
    min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    path_init_file = design_data_dict["gdp_init_file_path"]

    m = pyo.ConcreteModel()
    m.usc = usc_gdp.main(method=method,
                         max_power=max_power,
                         load_init_file=load_init_file,
                         path_init_file=path_init_file,
                         deact_arcs_after_init=deact_arcs_after_init,
                         solver=solver)


    # Set bounds for power produced by the plant alone
    m.usc.fs.plant_min_power_eq = pyo.Constraint(
        expr=m.usc.fs.plant_power_out[0] >= min_power
    )
    m.usc.fs.plant_max_power_eq = pyo.Constraint(
        expr=m.usc.fs.plant_power_out[0] <= max_power
    )

    # Set bounds in charge and discharge heat exchangers
    charge_mode = m.usc.fs.charge_mode_disjunct
    discharge_mode = m.usc.fs.discharge_mode_disjunct
    hxc_heat_duty = (1e-6) * (pyunits.MW / pyunits.W) * charge_mode.hxc.heat_duty[0]
    hxd_heat_duty = (1e-6) * (pyunits.MW / pyunits.W) * discharge_mode.hxd.heat_duty[0]
    m.usc.fs.charge_mode_disjunct.storage_heat_duty_lb = pyo.Constraint(
        expr=hxc_heat_duty >= min_storage_heat_duty + 40
    )
    m.usc.fs.discharge_mode_disjunct.storage_heat_duty_lb = pyo.Constraint(
        expr=hxd_heat_duty >= min_storage_heat_duty + 40
    )
    m.usc.fs.charge_mode_disjunct.storage_heat_duty_ub = pyo.Constraint(
        expr=hxc_heat_duty <= max_storage_heat_duty
    )
    m.usc.fs.discharge_mode_disjunct.storage_heat_duty_ub = pyo.Constraint(
        expr=hxd_heat_duty <= max_storage_heat_duty * (1 - 0.01)
    )

    # Unfix boiler data fixed during initialization
    m.usc.fs.boiler.inlet.flow_mol[0].unfix()

    if not deact_arcs_after_init:
        m.usc.fs.turbine[3].inlet.unfix()
        m.usc.fs.fwh[8].tube_inlet.unfix()

    # Unfix global variables fixed during initialization
    m.usc.fs.hx_pump_work.unfix()
    m.usc.fs.discharge_turbine_work.unfix()

    # Unfix storage system data. Note that the area of the charge and
    # discharge heat exchangers is unfixed and calculated during the
    # solution of the model.
    # m.usc.fs.fuel_cost.unfix()
    m.usc.fs.charge_mode_disjunct.ess_charge_split.split_fraction[0, "to_hxc"].unfix()
    m.usc.fs.discharge_mode_disjunct.ess_discharge_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [charge_mode.hxc]:
        salt_hxc.shell_inlet.unfix()
        salt_hxc.tube_inlet.flow_mass.unfix()
        salt_hxc.area.unfix()

    for salt_hxd in [discharge_mode.hxd]:
        salt_hxd.tube_inlet.unfix()
        salt_hxd.shell_inlet.flow_mass.unfix()
        salt_hxd.area.unfix()

    if not new_design:
        for unit in [charge_mode.cooler]:
            unit.inlet.unfix()
            m.usc.fs.charge_mode_disjunct.cooler.outlet.enth_mol[0].unfix()

    # Fix molten salt cold temperature in the discharge heat exchanger.
    # charge_mode.hxc.area.fix(hxc_area)
    # charge_mode.hxc.outlet_2.temperature[0].fix(hot_salt_temp)
    # discharge_mode.hxd.area.fix(hxd_area)
    # discharge_mode.hxd.shell_inlet.temperature[0].fix(hot_salt_temp)
    discharge_mode.hxd.shell_outlet.temperature[0].fix(cold_salt_temp)

    return m


def create_mp_block():
    """Create ultra-supercritical plant model and initialization for each
    time period

    """

    print('>>> Creating USC model and initialization for each time period')

    m = create_ss_model()
    b1 = m.usc

    # print('DOFs within mp create 1 =', degrees_of_freedom(m))

    # Add data for .json data file
    ramp_rate = design_data_dict["ramp_rate"]
    factor_mton = design_data_dict["factor_mton"] # factor for conversion kg to metric ton
    max_power_total = 700 # random high value

    # Add coupling variables
    b1.previous_power = pyo.Var(
        domain=NonNegativeReals,
        initialize=400,
        bounds=(min_power, max_power_total),
        doc="Previous period power in MW"
        )

    max_inventory = 1e7 * factor_mton # in mton
    min_inventory = 75000 * factor_mton # in mton
    max_salt_amount = design_data_dict["max_salt_amount"] * factor_mton # in mton
    tank_max = max_salt_amount

    b1.previous_salt_inventory_hot = pyo.Var(
        domain=NonNegativeReals,
        initialize=min_inventory,
        bounds=(0, max_inventory),
        doc="Hot salt at the beginning of the period in mton"
        )
    b1.salt_inventory_hot = pyo.Var(
        domain=NonNegativeReals,
        initialize=min_inventory,
        bounds=(0, max_inventory),
        doc="Hot salt inventory at the end of the period in mton"
        )
    b1.previous_salt_inventory_cold = pyo.Var(
        domain=NonNegativeReals,
        initialize=tank_max - min_inventory,
        bounds=(0, max_inventory),
        doc="Cold salt at the beginning of the period in mton"
        )
    b1.salt_inventory_cold = pyo.Var(
        domain=NonNegativeReals,
        initialize=tank_max - min_inventory,
        bounds=(0, max_inventory),
        doc="Cold salt inventory at the end of the in mton"
        )

    @b1.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (
            b1.previous_power - ramp_rate <=
            b.plant_power_out[0])

    @b1.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (
            b1.previous_power + ramp_rate >=
            b.plant_power_out[0])

    @b1.fs.Constraint(doc="Inventory balance at the end of the time period")
    def constraint_salt_inventory_hot(b):
        return (
            1e-3 * b1.salt_inventory_hot == (
                b1.previous_salt_inventory_hot
                + (3600 * b.salt_storage) * factor_mton # in mton
            ) * 1e-3
        )

    @b1.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            1e-3 * b.salt_amount == (
                b1.salt_inventory_hot
                + b1.salt_inventory_cold
            ) * 1e-3
        )

    # Scale variables and constraints
    # iscale.set_scaling_factor(b1.fs.fuel_cost, 1e-3)
    # iscale.set_scaling_factor(b1.fs.plant_fixed_operating_cost, 1e-3)
    # iscale.set_scaling_factor(b1.fs.plant_variable_operating_cost, 1e-3)
    # # iscale.set_scaling_factor(b1.fs.plant_capital_cost, 1e-3)

    iscale.set_scaling_factor(b1.fs.salt_amount, 1e-3)
    iscale.set_scaling_factor(b1.salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(b1.salt_inventory_cold, 1e-3)
    iscale.set_scaling_factor(b1.previous_salt_inventory_hot, 1e-3)
    iscale.set_scaling_factor(b1.previous_salt_inventory_cold, 1e-3)
    # iscale.set_scaling_factor(b1.fs.constraint_salt_inventory_hot, 1e-3)

    # iscale.set_scaling_factor(b1.fs.charge_mode_disjunct.capital_cost, 1e-3)
    # iscale.set_scaling_factor(b1.fs.discharge_mode_disjunct.capital_cost, 1e-3)
    # iscale.set_scaling_factor(b1.fs.storage_capital_cost, 1e-3)

    # Calculate scaling factors
    iscale.calculate_scaling_factors(m)

    return m


# The tank level and power output are linked between contiguous time
# periods.
def get_usc_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [
        (b1.usc.salt_inventory_hot, b2.usc.previous_salt_inventory_hot),
        (b1.usc.fs.plant_power_out[0], b2.usc.previous_power)
    ]


# The tank level at the end of the last period must be the same as the
# level at the beginning of the first period and power output must be
# the same as the initial tank level.
def get_usc_periodic_variable_pairs(b1, b2):
    """
        b1: final time block
        b2: first time block
    """

    return [(b1.usc.salt_inventory_hot, b2.usc.previous_salt_inventory_hot)]


# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.
def create_gdp_multiperiod_usc_model(n_time_points=None, pmin=None, pmax=None):
    """Create a multiperiod usc_mp cycle object. This object contains a
    Pyomo model with a block for each time instance

    n_time_points: Number of time blocks to create

    """

    multiperiod_usc = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=create_mp_block,
        linking_variable_func=get_usc_link_variable_pairs,
        periodic_variable_func=get_usc_periodic_variable_pairs
    )

    # If you have no arguments, you don't actually need to pass in
    # anything. NOTE: building the model will initialize each time block
    multiperiod_usc.build_multi_period_model()

    return multiperiod_usc
