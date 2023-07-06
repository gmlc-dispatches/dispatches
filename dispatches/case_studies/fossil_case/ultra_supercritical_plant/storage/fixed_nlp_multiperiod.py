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

__author__ = "Soraya Rawlings"


# Import Python libraries
import json
try:
    from importlib import resources  # Python 3.8+
except ImportError:
    import importlib_resources as resources  # Python 3.7


# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.environ import (Block, Param, Constraint, Objective, Reals,
                           NonNegativeReals, TransformationFactory, Expression,
                           maximize, RangeSet, value, log, exp, Var, ConcreteModel)
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.core.util.scaling import (list_unscaled_constraints,
                                     list_unscaled_variables,
                                     extreme_jacobian_rows,
                                     extreme_jacobian_columns)
from idaes.core.util.model_statistics import large_residuals_set

# Import IDAES libraries
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers.get_solver import get_solver
from idaes.core.util import from_json, to_json
import idaes.logger as idaeslog

# Import ultra-supercritical power plant model
from dispatches.case_studies.fossil_case.ultra_supercritical_plant import (
    ultra_supercritical_powerplant as usc)

use_surrogate = False
constant_salt = True
fix_design = True

# Import integrated ultrasupercritical power plant model. Also,
# include the data path for the model
print('>>>>> Solving for 4-12disj storage design')
data_path = 'fixed_uscp_design_data.json'
if use_surrogate:
    print('>>>>> Solving for new storage design using surrogate for USCPP')
    import usc_storage_nlp_mp_unfixed_area_new_storage_design_surrogate as usc_with_tes
else:
    print('>>>>> Solving for new storage design using rigorous model for USCPP')
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
    # m.max_inventory = pyo.units.convert(1e4*pyunits.tonne)
    m.max_inventory = pyo.units.convert(1e7*pyunits.kg,
                                        to_units=pyunits.metric_ton)
    # m.min_inventory = pyo.units.convert(100000*pyunits.kg,
    m.min_inventory = pyo.units.convert(200000*pyunits.kg,
                                        to_units=pyunits.metric_ton)
    m.tank_max = pyo.units.convert(design_data_dict["max_salt_amount"]*pyunits.kg,
                                   to_units=pyunits.metric_ton)
    m.hxc_area = design_data_dict["hxc_area"]
    m.hxd_area = design_data_dict["hxd_area"]

    # m.tank_min = 1e-3*pyunits.metric_ton
    m.tank_min = 0


def create_usc_model(m=None, pmin=None, pmax=None):

    optarg = {
        # "tol": 1e-8,
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Add options to integrated model
    method = "with_efficiency" # adds boiler and cycle efficiencies

    if m is None:
        m = pyo.ConcreteModel(name="Integrated Model")

    # Construct integrated flowsheet by calling functions to: (1)
    # Create integrated model, add properties, unit models, and arcs,
    # (2) Set required inputs to the model to have a square problem
    # for initialization, (3) add scaling factors, (4) add cost
    # correlations, and (5) add bounds
    # if use_surrogate:
    #     m = usc_with_tes.create_integrated_model(m, method=method)
    #     add_data(m)
    # else:
    # Add ultra-supercritical model
    m = usc.build_plant_model(m)

    add_data(m)
    m = usc_with_tes.create_integrated_model(m, method=method)

    usc_with_tes.set_model_input(m)
    usc_with_tes.set_scaling_factors(m)
    m = usc_with_tes.build_costing(m, optarg={"max_iter": 300})
    # usc_with_tes.add_bounds(m)

    # Set bounds for power produced by the plant without storage
    # m.fs.eq_turbine_temperature_out.deactivate()
    # m.fs.turbine_temperature_out_eq = pyo.Constraint(
    #     expr=(
    #         m.fs.es_turbine.control_volume.properties_out[0].temperature >=
    #         m.fs.es_turbine.control_volume.properties_out[0].temperature_sat + 1*pyunits.K
    #     )
    # )
    m.fs.plant_min_power_eq = pyo.Constraint(expr=m.fs.plant_power_out[0] >= m.pmin_plant)
    m.fs.plant_max_power_eq = pyo.Constraint(expr=m.fs.plant_power_out[0] <= m.pmax_plant)
    m.fs.net_plant_min_power_eq = pyo.Constraint(expr=m.fs.net_power[0] >= pmin)
    m.fs.net_plant_max_power_eq = pyo.Constraint(expr=m.fs.net_power[0] <= m.pmax_plant + m.pmax_storage)
    # m.fs.HX_T_equality_eq = pyo.Constraint(expr=m.fs.hxc.tube_outlet.temperature[0] == m.fs.hxd.shell_inlet.temperature[0])

    # Set lower and upper bounds to charge and discharge heat
    # exchangers
    m.fs.hxc_duty = pyo.units.convert(m.fs.hxc.heat_duty[0], to_units=pyunits.MW)
    m.fs.hxd_duty = pyo.units.convert(m.fs.hxd.heat_duty[0], to_units=pyunits.MW)
    m.fs.charge_storage_lb_eq = pyo.Constraint(expr=m.fs.hxc_duty >= m.min_storage_duty)
    m.fs.charge_storage_ub_eq = pyo.Constraint(expr=m.fs.hxc_duty <= m.max_storage_duty)
    m.fs.discharge_storage_lb_eq = pyo.Constraint(expr=m.fs.hxd_duty >= m.min_storage_duty)
    m.fs.discharge_storage_ub_eq = pyo.Constraint(expr=m.fs.hxd_duty <= m.max_storage_duty)
    # m.fs.charge_tube_outlet_T_lb_eq = pyo.Constraint(expr=m.fs.hxc.tube_outlet.temperature[0] >= 826)
    # m.fs.charge_delta_T_in_eq = pyo.Constraint(expr=m.fs.hxc.delta_temperature_in[0] >= 5)
    m.fs.discharge_delta_T_in_eq = pyo.Constraint(expr=m.fs.hxd.delta_temperature_in[0] >= 5)

    # Add coupling variables
    m.fs.previous_power = pyo.Var(domain=NonNegativeReals,
                                  initialize=436,
                                #   bounds=(pmin, pmax),
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
                                            #    bounds=(m.tank_min, m.max_inventory),
                                               units=pyunits.metric_ton,
                                               doc="Hot salt at the beginning of time period")
    m.fs.salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                      initialize=m.min_inventory,
                                    #   bounds=(m.tank_min, m.max_inventory),
                                      units=pyunits.metric_ton,
                                      doc="Hot salt inventory at the end of time period")
    m.fs.previous_salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                                initialize=m.fs.salt_amount - m.min_inventory,
                                                # bounds=(m.tank_min, m.max_inventory),
                                                units=pyunits.metric_ton,
                                                doc="Cold salt at the beginning of time period")
    m.fs.salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                       initialize=m.fs.salt_amount - m.min_inventory,
                                    #    bounds=(m.tank_min, m.max_inventory),
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
    # if not constant_salt:
    #     @m.fs.Constraint(doc="Hot salt inventory balance at the end of time period")
    #     def constraint_salt_inventory_cold(b):
    #         return (
    #             scaling_const*b.salt_inventory_cold == (
    #                 b.previous_salt_inventory_cold +
    #                 (b.hxd_flow_mass - b.hxc_flow_mass) # in metric_ton/h
    #             )*scaling_const
    #         )

    @m.fs.Constraint(doc="Maximum salt inventory at any time")
    def constraint_salt_inventory(b):
        return (
            scaling_const*b.salt_amount == (
                b.salt_inventory_hot +
                b.salt_inventory_cold
            )*scaling_const
        )

    # @m.fs.Constraint(doc="Maximum salt inventory at any time")
    # def constraint_previous_salt_inventory(b):
    #     return (
    #         scaling_const*b.salt_amount == (
    #             b.previous_salt_inventory_hot +
    #             b.previous_salt_inventory_cold
    #         )*scaling_const
    #     )

    @m.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return b.hxd_flow_mass <= b.previous_salt_inventory_hot

    @m.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return b.hxc_flow_mass <= b.previous_salt_inventory_cold

    return m


def usc_unfix_dof(m):

    # Unfix data fixed during during initialization
    # if use_surrogate:
    #     m.fs.boiler_flow.unfix()
    #     m.fs.steam_to_storage.unfix()
    #     m.fs.steam_to_discharge.unfix()
    # else:
    m.fs.boiler.inlet.flow_mol.unfix()
    # m.fs.ess_charge_split.split_fraction[0, "to_hxc"].unfix()
    # m.fs.ess_discharge_split.split_fraction[0, "to_hxd"].unfix()

    # if not constant_salt:
    #     m.fs.salt_amount.unfix()

    for charge_hxc in [m.fs.hxc]:
        charge_hxc.shell_inlet.unfix()
        charge_hxc.tube_inlet.flow_mass.unfix()
        charge_hxc.area.unfix()
        # charge_hxc.tube_outlet.temperature[0].unfix()

    for discharge_hxd in [m.fs.hxd]:
        discharge_hxd.tube_inlet.unfix()
        discharge_hxd.shell_inlet.flow_mass.unfix()
        discharge_hxd.area.unfix()
        # discharge_hxd.shell_inlet.temperature[0].unfix()

    # Unfix hx pump pressure
    m.fs.hx_pump.outlet.pressure[0].unfix()
    m.fs.previous_power.unfix()
    m.fs.previous_salt_inventory_cold.unfix()
    m.fs.previous_salt_inventory_hot.unfix()
    m.fs.salt_inventory_hot.unfix()
    # m.fs.linking_power_eq.deactivate()
    # # Fix storage heat exchangers area and salt temperatures
    # cold_salt_temperature = design_data_dict["cold_salt_temperature"]*pyunits.K
    # m.fs.hxd.shell_outlet.temperature[0].fix(cold_salt_temperature)

    # if fix_design:
    m.fs.hxc.area.fix(m.hxc_area)
    m.fs.hxd.area.fix(m.hxd_area)
    # # 828.596282
    # m.fs.hxc.tube_outlet.temperature[0].fix(826.56)
    # m.fs.hxd.shell_inlet.temperature[0].fix(826.56)
    #     # m.fs.charge_hot_salt_storage_lb_eq = pyo.Constraint(
    #     #     expr=m.fs.hxc.tube_outlet.temperature[0] >= 828
    #     # )
    #     # m.fs.charge_hot_salt_storage_ub_eq = pyo.Constraint(
    #     #     expr=m.fs.hxc.tube_outlet.temperature[0] <= 828.59
    #     # )
    #     # m.fs.discharge_hot_salt_storage_ub_eq = pyo.Constraint(
    #     #     expr=m.fs.hxd.shell_inlet.temperature[0] == m.fs.hxc.tube_outlet.temperature[0]
    #     # )


def usc_custom_init(m):

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Add options to integrated model
    method = "with_efficiency" # adds boiler and cycle efficiencies

    # Construct integrated flowsheet by calling functions to: (1)
    # Create ultra-supercritical model, (2) Create integrated plant
    # model and add its properties, unit models, and arcs, (2) Set
    # required inputs to the integrated model to have a square problem
    # for initialization, (3) add scaling factors, (4) initialize the
    # integrated model with a sequential initialization and custom
    # routines, (5) add cost correlations, and (6) initialize cost
    # equations.
    # if use_surrogate:
    #     blk = usc_with_tes.create_integrated_model(method=method)
    #     add_data(blk)
    # else:
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
    # usc_with_tes.add_bounds(blk)

    # Set lower and upper bounds to power produced by power plant and
    # charge and discharge storage heat duty
    blk.pmin = design_data_dict["plant_min_power"]*pyunits.MW
    blk.pmax = design_data_dict["plant_min_power"]*pyunits.MW
    blk.fs.plant_min_power_eq = pyo.Constraint(expr=blk.fs.plant_power_out[0] >= blk.pmin)
    blk.fs.plant_max_power_eq = pyo.Constraint(expr=blk.fs.plant_power_out[0] <= blk.pmax_plant)
    # blk.fs.net_plant_min_power_eq = pyo.Constraint(expr=blk.fs.net_power[0] >= blk.pmin)
    blk.fs.net_plant_max_power_eq = pyo.Constraint(expr=blk.fs.net_power[0] <= blk.pmax_plant + blk.pmax_storage)
    # blk.fs.eq_turbine_temperature_out.deactivate()
    # blk.fs.turbine_temperature_out_eq = pyo.Constraint(
    #     expr=(
    #         blk.fs.es_turbine.control_volume.properties_out[0].temperature >=
    #         blk.fs.es_turbine.control_volume.properties_out[0].temperature_sat + 1*pyunits.K
    #     )
    # )

    # Set lower and upper bounds to charge and discharge heat
    # exchangers
    blk.fs.hxc_duty = pyo.units.convert(blk.fs.hxc.heat_duty[0], to_units=pyunits.MW)
    blk.fs.hxd_duty = pyo.units.convert(blk.fs.hxd.heat_duty[0], to_units=pyunits.MW)
    blk.fs.charge_storage_lb_eq = pyo.Constraint(expr=blk.fs.hxc_duty >= blk.min_storage_duty)
    blk.fs.charge_storage_ub_eq = pyo.Constraint(expr=blk.fs.hxc_duty <= blk.max_storage_duty)
    blk.fs.discharge_storage_lb_eq = pyo.Constraint(expr=blk.fs.hxd_duty >= blk.min_storage_duty)
    blk.fs.discharge_storage_ub_eq = pyo.Constraint(expr=blk.fs.hxd_duty <= blk.max_storage_duty)
    # blk.fs.charge_tube_outlet_T_lb_eq = pyo.Constraint(expr=blk.fs.hxc.tube_outlet.temperature[0] >= 826)
    # blk.fs.charge_delta_T_in_eq = pyo.Constraint(expr=blk.fs.hxc.delta_temperature_in[0] >= 5)
    blk.fs.discharge_delta_T_in_eq = pyo.Constraint(expr=blk.fs.hxd.delta_temperature_in[0] >= 5)

    # Declare the plant power and salt inventory variables and
    # constraints
    blk.pmax = blk.pmax_plant + blk.pmax_storage
    blk.fs.previous_power = pyo.Var(domain=NonNegativeReals,
                                    initialize=425.8765,
                                    # bounds=(blk.pmin, blk.pmax),
                                    units=pyunits.MW,
                                    doc="Previous period power")
    blk.fs.previous_power.fix(426.29979551243326*pyunits.MW)
    # blk.fs.linking_power_eq = pyo.Constraint(expr=blk.fs.previous_power == blk.fs.plant_power_out[0.0])

    @blk.fs.Constraint(doc="Plant ramping down constraint")
    def constraint_ramp_down(b):
        return (b.previous_power - blk.ramp_rate) <= b.plant_power_out[0]

    @blk.fs.Constraint(doc="Plant ramping up constraint")
    def constraint_ramp_up(b):
        return (b.previous_power + blk.ramp_rate) >= b.plant_power_out[0]

    blk.fs.previous_salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                                 initialize=blk.min_inventory,
                                                #  bounds=(blk.tank_min, blk.max_inventory),
                                                 units=pyunits.metric_ton,
                                                 doc="Hot salt at the beginning of time period")
    blk.fs.salt_inventory_hot = pyo.Var(domain=NonNegativeReals,
                                        initialize=blk.min_inventory,
                                        # bounds=(blk.tank_min, blk.max_inventory),
                                        units=pyunits.metric_ton,
                                        doc="Hot salt inventory at the end of time period")
    blk.fs.previous_salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                                  initialize=blk.fs.salt_amount - blk.min_inventory,
                                                #   bounds=(blk.tank_min, blk.max_inventory),
                                                  units=pyunits.metric_ton,
                                                  doc="Cold salt at the beginning of time period")
    blk.fs.salt_inventory_cold = pyo.Var(domain=NonNegativeReals,
                                         initialize=blk.fs.salt_amount - blk.min_inventory,
                                        #  bounds=(blk.tank_min, blk.max_inventory),
                                         units=pyunits.metric_ton,
                                         doc="Cold salt inventory at the end of time period")

    blk.fs.hxc_flow_mass = pyo.units.convert(blk.fs.hxc.tube_inlet.flow_mass[0],
                                             to_units=pyunits.metric_ton/pyunits.hour)
    blk.fs.hxd_flow_mass = pyo.units.convert(blk.fs.hxd.shell_inlet.flow_mass[0],
                                             to_units=pyunits.metric_ton/pyunits.hour)
    blk.fs.previous_salt_inventory_cold.fix()
    blk.fs.previous_salt_inventory_hot.fix()
    # blk.fs.hxd.shell_inlet.flow_mass.unfix()
    # blk.fs.salt_inventory_hot.fix()
    @blk.fs.Constraint(doc="Inventory balance at the end of time period")
    def constraint_salt_inventory_hot(b):
        return (
            scaling_const*b.salt_inventory_hot == (
                b.previous_salt_inventory_hot +
                (b.hxc_flow_mass - b.hxd_flow_mass) # in metric_ton/hour
            )*scaling_const
        )
    # calculate_variable_from_constraint(blk.fs.salt_inventory_hot,
    #                                    blk.fs.constraint_salt_inventory_hot)

    # if not constant_salt:
    #     @blk.fs.Constraint(doc="Inventory balance at the end of time period")
    #     def constraint_salt_inventory_cold(b):
    #         return (
    #             scaling_const*b.salt_inventory_cold == (
    #                 b.previous_salt_inventory_cold +
    #                 (b.hxd_flow_mass - b.hxc_flow_mass) # in metric_ton/hour
    #             )*scaling_const
    #         )

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

    # @blk.fs.Constraint(doc="Maximum salt inventory at any time")
    # def constraint_previous_salt_inventory(b):
    #     return (
    #         scaling_const*b.salt_amount == (
    #             b.previous_salt_inventory_hot +
    #             b.previous_salt_inventory_cold
    #         )*scaling_const
    #     )

    # @blk.Constraint()
    # def rule_periodic_variable_pair(b):
    #     return (b.fs.salt_inventory_hot ==
    #             b.fs.previous_salt_inventory_hot)

    @blk.fs.Constraint(doc="Max salt flow to hxd based on available hot salt")
    def constraint_salt_maxflow_hot(b):
        return b.hxd_flow_mass <= b.previous_salt_inventory_hot

    @blk.fs.Constraint(doc="Max salt flow to hxc based on available cold salt")
    def constraint_salt_maxflow_cold(b):
        return b.hxc_flow_mass <= b.previous_salt_inventory_cold

    solver.solve(blk, tee=True)
    print("  ")
    print("  ")
    print("I WAS HERE")
    print("Hot salt level", value(blk.fs.salt_inventory_hot))
    print("Previous Hot salt level", value(blk.fs.previous_salt_inventory_hot))
    print("Cold salt level", value(blk.fs.salt_inventory_cold))
    print("Previous Cold salt level", value(blk.fs.previous_salt_inventory_cold))
    print("Plant Power", value(blk.fs.plant_power_out[0.0]))
    print("Previous plant Power", value(blk.fs.previous_power))
    print("  ")
    print("  ")
    print("Print out all constraints with residuals larger than 0.1")
    lrs = large_residuals_set(blk)
    for i in lrs:
        print(i.name)
    print("  ")
    print("  ")
    # assert False
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
        # (b1.fs.salt_inventory_cold, b2.fs.previous_salt_inventory_cold),
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
