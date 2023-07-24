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
state NLP multiperiod model for an integrated ultra-supercritical
power plant model. The purpose of this script is to create an NLP
multiperiod model that can be use for market analysis using a
pricetaker assumption. The integrated storage with ultra-supercritical
power plant model is used a steady state model for creating the
multiperiod model.

"""

__author__ = "Soraya Rawlings and Naresh Susarla"


# Import Python libraries
import json
try:
    from importlib import resources  # Python 3.8+
except ImportError:
    import importlib_resources as resources  # Python 3.7


# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.environ import NonNegativeReals
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES libraries
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util import from_json, to_json
import idaes.logger as idaeslog

# Import ultra-supercritical power plant model
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

# Import integrated ultrasupercritical power plant model. Also,
# include the data path for the model
data_path = 'fixed_uscp_design_data.json'
print('>>>>> Solving for storage design obtained from superstructure optimization')
import fixed_integrated_usc_storage as usc_with_tes

with open(data_path) as design_data:
    design_data_dict = json.load(design_data)

scaling_const = 1e-3

def add_data(m):
    # Add data from .json file
    m.pmin_plant = design_data_dict["plant_min_power"]*pyunits.MW
    m.pmax_plant = design_data_dict["plant_max_power"]*pyunits.MW
    m.pmax_storage = design_data_dict["max_discharge_turbine_power"]*pyunits.MW
    m.min_storage_duty = design_data_dict["min_storage_duty"]*pyunits.MW
    m.max_storage_duty = design_data_dict["max_storage_duty"]*pyunits.MW
    m.ramp_rate = design_data_dict["ramp_rate"]*pyunits.MW
    m.max_inventory = pyo.units.convert(1e7*pyunits.kg,
                                        to_units=pyunits.metric_ton)
    m.min_inventory = pyo.units.convert(200000*pyunits.kg,
                                        to_units=pyunits.metric_ton)
    m.tank_max = pyo.units.convert(design_data_dict["max_salt_amount"]*pyunits.kg,
                                   to_units=pyunits.metric_ton)
    m.hxc_area = design_data_dict["hxc_area"]
    m.hxd_area = design_data_dict["hxd_area"]

    m.tank_min = 0


def create_usc_model(m=None, pmin=None, pmax=None):

    optarg = {
        "max_iter": 300,
    }
    solver = get_solver('ipopt', optarg)

    # Add options to integrated model
    method = "with_efficiency" # adds boiler and cycle efficiencies

    if m is None:
        m = pyo.ConcreteModel(name="Integrated Model")

    # Add ultra-supercritical model
    m = usc.build_plant_model(m)

    add_data(m)
    m = usc_with_tes.create_integrated_model(m, method=method)

    usc_with_tes.set_model_input(m)
    usc_with_tes.set_scaling_factors(m)
    m = usc_with_tes.build_costing(m, optarg={"max_iter": 300})

    m.fs.plant_min_power_eq = pyo.Constraint(expr=m.fs.plant_power_out[0] >= m.pmin_plant)
    m.fs.plant_max_power_eq = pyo.Constraint(expr=m.fs.plant_power_out[0] <= m.pmax_plant)
    m.fs.net_plant_min_power_eq = pyo.Constraint(expr=m.fs.net_power[0] >= pmin)
    m.fs.net_plant_max_power_eq = pyo.Constraint(expr=m.fs.net_power[0] <= m.pmax_plant + m.pmax_storage)

    # Set lower and upper bounds to charge and discharge heat
    # exchangers
    m.fs.hxc_duty = pyo.units.convert(m.fs.hxc.heat_duty[0], to_units=pyunits.MW)
    m.fs.hxd_duty = pyo.units.convert(m.fs.hxd.heat_duty[0], to_units=pyunits.MW)
    m.fs.charge_storage_lb_eq = pyo.Constraint(expr=m.fs.hxc_duty >= m.min_storage_duty)
    m.fs.charge_storage_ub_eq = pyo.Constraint(expr=m.fs.hxc_duty <= m.max_storage_duty)
    m.fs.discharge_storage_lb_eq = pyo.Constraint(expr=m.fs.hxd_duty >= m.min_storage_duty)
    m.fs.discharge_storage_ub_eq = pyo.Constraint(expr=m.fs.hxd_duty <= m.max_storage_duty)
    m.fs.discharge_delta_T_in_eq = pyo.Constraint(expr=m.fs.hxd.delta_temperature_in[0] >= 5)

    # Add coupling variables
    m.fs.previous_power = pyo.Var(domain=NonNegativeReals,
                                  initialize=436,
                                  units=pyunits.MW,
                                  doc="Previous period power")
    @m.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (b.previous_power - m.ramp_rate) <= b.plant_power_out[0]

    @m.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (b.previous_power + m.ramp_rate) >= b.plant_power_out[0]

    m.fs.previous_salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                               initialize=m.min_inventory,
                                               units=pyunits.metric_ton,
                                               doc="Hot salt at the beginning of time period")
    m.fs.salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                      initialize=m.min_inventory,
                                      units=pyunits.metric_ton,
                                      doc="Hot salt inventory at the end of time period")
    m.fs.previous_salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                                initialize=m.fs.salt_amount - m.min_inventory,
                                                units=pyunits.metric_ton,
                                                doc="Cold salt at the beginning of time period")
    m.fs.salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                       initialize=m.fs.salt_amount - m.min_inventory,
                                       units=pyunits.metric_ton,
                                       doc="Cold salt inventory at the end of time period")

    m.fs.hxc_flow_mass = pyo.units.convert(m.fs.hxc.tube_inlet.flow_mass[0],
                                           to_units=pyunits.metric_ton/pyunits.hour)
    m.fs.hxd_flow_mass = pyo.units.convert(m.fs.hxd.shell_inlet.flow_mass[0],
                                           to_units=pyunits.metric_ton/pyunits.hour)
    @m.fs.Constraint(doc="Hot salt inventory balance at the end of time period")
    def constraint_salt_inventory_hot(b):
        return (
            scaling_const*b.salt_inventory_hot == (
                b.previous_salt_inventory_hot +
                (b.hxc_flow_mass - b.hxd_flow_mass) # in metric_ton/h
            )*scaling_const
        )

    @m.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            scaling_const*b.salt_amount == (
                b.salt_inventory_hot +
                b.salt_inventory_cold
            )*scaling_const
        )


    @m.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return b.hxd_flow_mass <= b.previous_salt_inventory_hot

    @m.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return b.hxc_flow_mass <= b.previous_salt_inventory_cold

    return m


def usc_unfix_dof(m):

    # Unfix data fixed during during initialization
    m.fs.boiler.inlet.flow_mol.unfix()

    for charge_hxc in [m.fs.hxc]:
        charge_hxc.shell_inlet.unfix()
        charge_hxc.tube_inlet.flow_mass.unfix()
        charge_hxc.area.unfix()

    for discharge_hxd in [m.fs.hxd]:
        discharge_hxd.tube_inlet.unfix()
        discharge_hxd.shell_inlet.flow_mass.unfix()
        discharge_hxd.area.unfix()

    # Unfix hx pump pressure
    m.fs.hx_pump.outlet.pressure[0].unfix()
    m.fs.previous_power.unfix()
    m.fs.previous_salt_inventory_cold.unfix()
    m.fs.previous_salt_inventory_hot.unfix()
    m.fs.salt_inventory_hot.unfix()

    m.fs.hxc.area.fix(m.hxc_area)
    m.fs.hxd.area.fix(m.hxd_area)


def usc_custom_init(m):

    optarg = {
        "max_iter": 300,
    }
    solver = get_solver('ipopt', optarg)

    # Add options to integrated model
    method = "with_efficiency" # adds boiler and cycle efficiencies

    blk = usc.build_plant_model()
    usc.initialize(blk)

    # Add data from .json file
    add_data(blk)

    blk = usc_with_tes.create_integrated_model(blk, method=method)

    usc_with_tes.set_model_input(blk)
    usc_with_tes.set_scaling_factors(blk)
    usc_with_tes.initialize(blk)
    blk = usc_with_tes.build_costing(blk, optarg={"max_iter": 300})
    usc_with_tes.initialize_with_costing(blk, solver=solver)

    # Set lower and upper bounds to power produced by power plant and
    # charge and discharge storage heat duty
    blk.pmin = design_data_dict["plant_min_power"]*pyunits.MW
    blk.pmax = design_data_dict["plant_min_power"]*pyunits.MW
    blk.fs.plant_min_power_eq = pyo.Constraint(expr=blk.fs.plant_power_out[0] >= blk.pmin)
    blk.fs.plant_max_power_eq = pyo.Constraint(expr=blk.fs.plant_power_out[0] <= blk.pmax_plant)
    blk.fs.net_plant_max_power_eq = pyo.Constraint(expr=blk.fs.net_power[0] <= blk.pmax_plant + blk.pmax_storage)

    # Set lower and upper bounds to charge and discharge heat
    # exchangers
    blk.fs.hxc_duty = pyo.units.convert(blk.fs.hxc.heat_duty[0], to_units=pyunits.MW)
    blk.fs.hxd_duty = pyo.units.convert(blk.fs.hxd.heat_duty[0], to_units=pyunits.MW)
    blk.fs.charge_storage_lb_eq = pyo.Constraint(expr=blk.fs.hxc_duty >= blk.min_storage_duty)
    blk.fs.charge_storage_ub_eq = pyo.Constraint(expr=blk.fs.hxc_duty <= blk.max_storage_duty)
    blk.fs.discharge_storage_lb_eq = pyo.Constraint(expr=blk.fs.hxd_duty >= blk.min_storage_duty)
    blk.fs.discharge_storage_ub_eq = pyo.Constraint(expr=blk.fs.hxd_duty <= blk.max_storage_duty)
    blk.fs.discharge_delta_T_in_eq = pyo.Constraint(expr=blk.fs.hxd.delta_temperature_in[0] >= 5)

    # Declare the plant power and salt inventory variables and
    # constraints
    blk.pmax = blk.pmax_plant + blk.pmax_storage
    blk.fs.previous_power = pyo.Var(domain=NonNegativeReals,
                                    initialize=425.8765,
                                    units=pyunits.MW,
                                    doc="Previous period power")
    blk.fs.previous_power.fix(426.29979551243326*pyunits.MW)

    @blk.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (b.previous_power - blk.ramp_rate) <= b.plant_power_out[0]

    @blk.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (b.previous_power + blk.ramp_rate) >= b.plant_power_out[0]

    blk.fs.previous_salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                                 initialize=blk.min_inventory,
                                                 units=pyunits.metric_ton,
                                                 doc="Hot salt at the beginning of time period")
    blk.fs.salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                        initialize=blk.min_inventory,
                                        units=pyunits.metric_ton,
                                        doc="Hot salt inventory at the end of time period")
    blk.fs.previous_salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                                  initialize=blk.fs.salt_amount - blk.min_inventory,
                                                  units=pyunits.metric_ton,
                                                  doc="Cold salt at the beginning of time period")
    blk.fs.salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                         initialize=blk.fs.salt_amount - blk.min_inventory,
                                         units=pyunits.metric_ton,
                                         doc="Cold salt inventory at the end of time period")

    blk.fs.hxc_flow_mass = pyo.units.convert(blk.fs.hxc.tube_inlet.flow_mass[0],
                                             to_units=pyunits.metric_ton/pyunits.hour)
    blk.fs.hxd_flow_mass = pyo.units.convert(blk.fs.hxd.shell_inlet.flow_mass[0],
                                             to_units=pyunits.metric_ton/pyunits.hour)
    blk.fs.previous_salt_inventory_cold.fix()
    blk.fs.previous_salt_inventory_hot.fix()
    @blk.fs.Constraint(doc="Inventory balance at the end of time period")
    def constraint_salt_inventory_hot(b):
        return (
            scaling_const*b.salt_inventory_hot == (
                b.previous_salt_inventory_hot +
                (b.hxc_flow_mass - b.hxd_flow_mass) # in metric_ton/hour
            )*scaling_const
        )

    @blk.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            scaling_const*b.salt_amount == (
                b.salt_inventory_hot +
                b.salt_inventory_cold
            )*scaling_const
        )
    calculate_variable_from_constraint(blk.fs.salt_inventory_cold,
                                       blk.fs.constraint_salt_inventory)

    @blk.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return b.hxd_flow_mass <= b.previous_salt_inventory_hot

    @blk.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return b.hxc_flow_mass <= b.previous_salt_inventory_cold

    solver.solve(blk)
    init_model = to_json(blk, return_dict=True)
    from_json(m, sd=init_model)
    return

# The tank level and power output are linked between contiguous time
# periods
def get_usc_link_variable_pairs(b1, b2):
    """
        b1: current time block
        b2: next time block
    """
    return [
        (b1.fs.salt_inventory_hot, b2.fs.previous_salt_inventory_hot),
        (b1.fs.plant_power_out[0], b2.fs.previous_power)
    ]


# Create the multiperiod model object. You can pass arguments to your
# "process_model_func" for each time period using a dict of dicts as
# shown here.  In this case, it is setting up empty dictionaries for
# each time period.
def create_nlp_multiperiod_usc_model(n_time_points=None, pmin=None, pmax=None):
    """Create a multiperiod usc_mp cycle object. This object contains a
    Pyomo model with a block for each time instance

    n_time_points: Number of time blocks to create

    """

    multiperiod_usc = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=create_usc_model,
        initialization_func=usc_custom_init,
        unfix_dof_func=usc_unfix_dof,
        linking_variable_func=get_usc_link_variable_pairs,
        flowsheet_options={"pmin": pmin,
                           "pmax": pmax},
        use_stochastic_build=True,
        outlvl=idaeslog.INFO
    )

    return multiperiod_usc
