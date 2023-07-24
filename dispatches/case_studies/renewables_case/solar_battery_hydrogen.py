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
from pathlib import Path
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.case_studies.renewables_case.RE_flowsheet import create_model, propagate_state, value, h2_mols_per_kg, PA, discount_rate, N
from dispatches.case_studies.renewables_case.load_parameters import kg_to_tons
from dispatches.case_studies.renewables_case.solar_battery_hydrogen_inputs import (re_h2_parameters, h2_blend_ratio, s_per_ts, mmbtu_to_ng_kg, timestep_hrs, co2_emissions_lb_mmbtu, 
                                           capacity_credit_battery, capacity_requirement)


def pv_battery_hydrogen_variable_pairs(m1, m2):
    """
    This function links together unit model state variables from one timestep to the next.

    The simple hydrogen tank and the battery model have material and energy holdups that need to be consistent across time blocks.

    Args:
        m1: current time block model
        m2: next time block model
    """
    pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput),
              (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    
    return pairs


def pv_battery_hydrogen_periodic_variable_pairs(m1, m2):
    """
    The final hydrogen material holdup and battery storage of charge must be the same as in the intial timestep. 

    Args:
        m1: final time block model
        m2: first time block model
    """
    pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    return pairs


def initialize_fs(m, input_params=dict(), verbose=False):
    """
    Initializing the flowsheet is done starting with the pv model and propagating the solved initial state to downstream models.

    The splitter is initialized with no flow to the battery or PEM so all electricity flows to the grid, which makes the initialization of all
    unit models downstream of the pv plant independent of its time-varying electricity production. This initialzation function can
    then be repeated for all timesteps within a dynamic analysis.

    Args:
        m: model
        verbose:
    """
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING

    m.fs.pv.initialize(outlvl=outlvl)

    propagate_state(m.fs.pv_to_splitter)
    m.fs.splitter.battery_elec[0].fix(0)
    m.fs.splitter.pem_elec[0].fix(0)
    m.fs.splitter.initialize()
    m.fs.splitter.battery_elec[0].unfix()
    m.fs.splitter.pem_elec[0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_pem)
    propagate_state(m.fs.splitter_to_battery)

    batt_init = input_params['batt_mw'] * 1e3
    if 'batt_soc_init_mwh' in input_params.keys():
        batt_init = input_params['batt_soc_init_mwh'] * 1e3
    m.fs.battery.initial_state_of_charge.fix(batt_init)
    m.fs.battery.initial_energy_throughput.fix(batt_init)
    m.fs.battery.elec_in[0].fix()
    m.fs.battery.elec_out[0].fix(value(m.fs.battery.elec_in[0]))
    m.fs.battery.initialize(outlvl=outlvl)
    m.fs.battery.elec_in[0].unfix()
    m.fs.battery.elec_out[0].unfix()
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()
    if verbose:
        m.fs.battery.report(dof=True)

    m.fs.pem.initialize(outlvl=outlvl)
    if verbose:
        m.fs.pem.report(dof=True)

    propagate_state(m.fs.pem_to_tank)

    tank_holdup_init = input_params['turbine_min_mw'] * 1e3 * h2_blend_ratio / input_params['ng_turb_conv'] * h2_mols_per_kg
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix(0)
    m.fs.h2_tank.tank_holdup_previous.fix(tank_holdup_init)
    m.fs.h2_tank.initialize(outlvl=outlvl)
    m.fs.h2_tank.outlet_to_turbine.flow_mol[0].unfix()
    m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].unfix()
    m.fs.h2_tank.tank_holdup_previous.unfix()


def pv_battery_hydrogen_model(pv_resource_config, input_params, verbose):
    """
    Creates an initialized flowsheet model for a single time step with operating, size and cost components
    
    First, the model is created using the input_params and pv_resource_config
    Second, the model is initialized so that it solves and its values are internally consistent
    Third, battery ramp constraints and operating cost components are added

    Args:
        pvd_resource_config: PV resource for the time step
        input_params: size and operation parameters. Required keys: `pv_mw`, `pem_bar`, `batt_mw`, `tank_size`, `pem_bar`, 
        `ng_turb_conv`, h2_turb_conv
        verbose:
    """
    m = create_model(input_params['pv_mw'], input_params['pem_bar'], input_params['batt_mw'], "simple", input_params['tank_size'], None,
                     pv_resource_config, re_type='pv')
    if 'batt_hr' in input_params.keys():
        input_params['batt_mwh'] = input_params['batt_mw'] * input_params['batt_hr']
    if 'batt_mwh' in input_params.keys():
        m.fs.battery.nameplate_energy.fix(input_params['batt_mwh'] * 1e3)

    # battery between 0.5 to 8 hr
    m.fs.battery.four_hr_battery.deactivate()

    m.fs.h2_kg = pyo.Expression(expr=m.fs.h2_tank.outlet_to_turbine.flow_mol[0] * s_per_ts / h2_mols_per_kg)
    if h2_blend_ratio == 1:
        m.fs.ng_kg = pyo.Param(domain=pyo.NonNegativeReals, initialize=0)
    elif h2_blend_ratio != 0:
        m.fs.ng_kg = pyo.Expression(expr=m.fs.h2_kg / h2_blend_ratio - m.fs.h2_kg)
    else:
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(0)
        m.fs.ng_kg = pyo.Var(domain=pyo.NonNegativeReals, initialize=0)

    m.fs.turbine_ng_elec = pyo.Expression(expr=m.fs.ng_kg * input_params['ng_turb_conv'])
    m.fs.turbine_h2_elec = pyo.Expression(expr=m.fs.h2_kg * input_params['h2_turb_conv'])
    m.fs.turbine_elec_total = pyo.Expression(expr=m.fs.turbine_ng_elec + m.fs.turbine_h2_elec)
    m.fs.h2_turbine_pmin = pyo.Constraint(expr=m.fs.turbine_elec_total >= input_params['turbine_min_mw'] * 1e3)

    initialize_fs(m, input_params, verbose=verbose)

    # unfix for design optimization
    if input_params['design_opt']:
        m.fs.battery.nameplate_power.unfix()
        m.fs.battery.nameplate_energy.unfix()
        m.fs.pv.system_capacity.unfix()
    else:
        m.fs.pv.system_capacity.fix(input_params['pv_mw'] * 1e3)
        m.fs.battery.nameplate_power.fix(input_params['batt_mw'] * 1e3)
        m.fs.battery.nameplate_energy.fix(input_params['batt_mwh'] * 1e3)

    m.fs.battery.degradation_rate.set_value(0)
    return m


def pv_battery_hydrogen_mp_block(pv_resource_config, input_params, verbose):
    """
    Wrapper of `pv_battery_hydrogen_model` for creating the process model per time point for the MultiPeriod model.
    Uses cloning of the Pyomo model in order to reduce runtime. 
    The clone is reinitialized with the `pv_resource_config` for the given time point, which only required modifying
    the pvpower and the splitter, as the rest of the units have no flow and therefore is unaffected by pv resource changes.

    Args:
        pv_resource_config: dictionary with `resource_speed` for the time step
        input_params: size and operation parameters. Required keys: `pv_mw`, `pem_bar`, `batt_mw`, `tank_size`, `pem_bar`, 
            `ng_turb_conv`, h2_turb_conv
        verbose:
    """

    if 'pyo_model' not in input_params.keys():
        input_params['pyo_model'] = pv_battery_hydrogen_model(pv_resource_config, input_params, verbose)
    m = input_params['pyo_model'].clone()

    m.fs.pv.config.capacity_factor = pv_resource_config['capacity_factor']

    m.fs.pv.setup_resource()

    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
    m.fs.pv.initialize(outlvl=outlvl)
    propagate_state(m.fs.pv_to_splitter)
    m.fs.splitter.initialize()
    return m


def size_constraints(mp_model, input_params):
    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    m.pv_system_capacity = pyo.Param(default=input_params['pv_mw'] * 1e3, units=pyo.units.kW)
    m.pv_add_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW, bounds=(0, 1e7))
    m.battery_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['batt_mw'] * 1e3, units=pyo.units.kW, bounds=(0, 1e7))
    m.battery_system_energy = pyo.Var(domain=pyo.NonNegativeReals, initialize=(input_params['batt_mwh'] if 'batt_mwh' in input_params.keys()
                                                                                else (input_params['batt_mw'] * input_params['batt_hr'])) * 1e3, units=pyo.units.kWh)
    m.pem_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['pem_mw'] * 1e3, units=pyo.units.kW, bounds=(0, 1e7))
    m.h2_tank_size = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['tank_size'], units=pyo.units.kg, bounds=(0, 1e7))
    m.turb_system_capacity = pyo.Var(domain=pyo.NonNegativeReals, initialize=input_params['turb_mw'] * 1e3, units=pyo.units.kW, bounds=(input_params['turb_mw'] * 1e3, 1e8))

    if not input_params['design_opt']:
        m.pv_add_system_capacity.fix()
        m.battery_system_capacity.fix()
        m.battery_system_energy.fix()
        m.pem_system_capacity.fix()
        m.h2_tank_size.fix()
        m.turb_system_capacity.fix()

    m.pv_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.pv.system_capacity <= m.pv_system_capacity + m.pv_add_system_capacity)
    m.battery_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_power <= m.battery_system_capacity)
    m.battery_max_e = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_energy <= m.battery_system_energy)
    m.pem_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)
    m.tank_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.h2_tank.tank_holdup[0] / h2_mols_per_kg <= m.h2_tank_size)
    m.turb_max_p = pyo.Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.turbine_elec_total <= m.turb_system_capacity)

    # battery between 0.5 to 8 hr
    m.battery_min_hr = pyo.Constraint(expr=m.battery_system_energy >= m.battery_system_capacity * 0.5)
    m.battery_max_hr = pyo.Constraint(expr=m.battery_system_energy <= m.battery_system_capacity * 8)

def calculate_capital_costs(m, input_params):
    """
    Does not include existing PV and Turbine capacity into capital cost
    """
    # capital costs
    m.pv_cap_cost = pyo.Param(default=input_params["pv_cap_cost"], mutable=True)
    m.pem_cap_cost = pyo.Param(default=input_params["pem_cap_cost"], mutable=True)
    m.batt_cap_cost_kw = pyo.Param(default=input_params["batt_cap_cost_kw"], mutable=True)
    m.batt_cap_cost_kwh = pyo.Param(default=input_params["batt_cap_cost_kwh"], mutable=True)
    m.tank_cap_cost = pyo.Param(default=input_params["tank_cap_cost_per_kg"], mutable=True)
    m.turb_cap_cost = pyo.Param(default=input_params["turbine_cap_cost"], mutable=True)

    m.total_cap_cost = pyo.Expression(expr=m.pv_cap_cost * m.pv_add_system_capacity
                                       + m.batt_cap_cost_kw * m.battery_system_capacity
                                       + m.batt_cap_cost_kwh * m.battery_system_energy
                                       + m.pem_cap_cost * m.pem_system_capacity
                                       + m.tank_cap_cost * m.h2_tank_size
                                       + m.turb_cap_cost * (m.turb_system_capacity - input_params['turb_mw'] * 1e3))


def calculate_fixed_costs(m, input_params):
    m.pv_op_cost_unit = pyo.Param(
        initialize=input_params["pv_op_cost"],
        doc="fixed cost of operating pv plant $/kW-yr")
    m.pem_op_cost_unit = pyo.Param(
        initialize=input_params["pem_op_cost"],
        doc="fixed cost of operating pem $/kW-yr")
    m.h2_tank_op_cost_unit = pyo.Param(
        initialize=input_params["tank_op_cost"],
        doc="fixed cost of operating tank in $/kg-yr")
    m.h2_turbine_op_cost_unit = pyo.Param(
        initialize=input_params["turbine_op_cost"],
        doc="fixed cost of operating turbine in $/kW-yr")

    m.annual_fixed_cost = pyo.Expression(expr=(m.pv_system_capacity + m.pv_add_system_capacity) * m.pv_op_cost_unit
                                              + m.pem_system_capacity * m.pem_op_cost_unit
                                              + m.h2_tank_size * m.h2_tank_op_cost_unit
                                              + m.turb_system_capacity * m.h2_turbine_op_cost_unit)


def calculate_variable_costs(mp_model, input_params):
    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    m.battery_var_cost_unit = pyo.Param(
        initialize=input_params["batt_rep_cost_kwh"],
        doc="variable cost of battery degradation $/kwH")
    m.pem_var_cost_unit = pyo.Param(
        initialize=input_params["pem_var_cost"],
        doc="variable operating cost of pem $/kWh")
    m.h2_turbine_var_cost_unit = pyo.Param(
        initialize=input_params["turbine_var_cost"],
        doc="variable cost of operating turbine in $/kWh")

    for blk in blks:
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem

        blk_battery.var_cost = pyo.Expression(
            expr=blk_battery.degradation_rate * (blk_battery.energy_throughput[0] - blk_battery.initial_energy_throughput) * m.battery_var_cost_unit)
        blk_pem.var_cost = pyo.Expression(
            expr=m.pem_var_cost_unit * blk_pem.electricity[0])
        blk.turb_var_cost = pyo.Expression(
            expr=m.h2_turbine_var_cost_unit * blk.fs.turbine_elec_total
        )
        blk.var_total_cost = pyo.Expression(expr=blk_pem.var_cost
                                                 + blk_battery.var_cost
                                                 + blk.turb_var_cost)


def add_load_following_obj(mp_model, input_params):
    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()
    n_weeks = len(blks) / (7 * 24)

    prev_turb_elec = blks[-1].fs.turbine_elec_total
    for (i, blk) in enumerate(blks):
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem
        blk_tank = blk.fs.h2_tank
        blk_pv = blk.fs.pv
        turb_elec = blk.fs.turbine_elec_total

        # ramp constraints
        turb_elec.energy_down_ramp = pyo.Constraint(expr=prev_turb_elec - turb_elec <= input_params['turbine_ramp_mw_per_min'] * 1e3)
        turb_elec.energy_up_ramp = pyo.Constraint(expr=turb_elec - prev_turb_elec <= input_params['turbine_ramp_mw_per_min'] * 1e3)

        # meet load
        blk.load_power = pyo.Param(default=input_params['load'][i] * 1e3, mutable=True, units=pyo.units.kW)   # convert to kW
        blk.output_power = pyo.Expression(expr=blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0] + turb_elec)
        blk.grid_purchase = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)
        blk.grid_sales = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)
        blk.meet_load = pyo.Constraint(expr=blk.output_power + blk.grid_purchase == blk.load_power + blk.grid_sales)

        # meet reserve
        blk.excess_pv = pyo.Expression(expr=blk_pv.system_capacity * blk_pv.capacity_factor[0] - blk_pv.electricity[0])
        blk.h2_kg_reserve = pyo.Expression(expr=blk_tank.tank_holdup[0] / h2_mols_per_kg)
        blk.turbine_reserve = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)
        if h2_blend_ratio > 0:
            blk.ng_kg_reserve = pyo.Expression(expr=blk.h2_kg_reserve * (1 / h2_blend_ratio - 1))
            blk.fuel_reserve = pyo.Expression(expr=(blk.h2_kg_reserve * input_params['h2_turb_conv']
                                                    + blk.ng_kg_reserve * input_params['ng_turb_conv']))
            blk.turbine_reserve_lb1 = pyo.Constraint(expr=blk.turbine_reserve <= blk.fuel_reserve)
        blk.turbine_reserve_lb2 = pyo.Constraint(expr=blk.turbine_reserve <= m.turb_system_capacity - turb_elec)
        blk.battery_reserve = pyo.Var(domain=pyo.NonNegativeReals, initialize=0, units=pyo.units.kW)
        blk.battery_reserve_lb1 = pyo.Constraint(expr=blk.battery_reserve <= m.battery_system_capacity)
        blk.battery_reserve_lb2 = pyo.Constraint(expr=blk.battery_reserve <= blk_battery.state_of_charge[0])
        blk.pem_reserve = pyo.Expression(expr=blk_pem.electricity[0])
        blk.total_reserve = pyo.Expression(expr=blk.battery_reserve + blk.turbine_reserve + blk.excess_pv + blk.pem_reserve)
        blk.reserve_over_1hr = pyo.Param(default=(max(input_params['reserve'][max(i - int(1/timestep_hrs), 0):i]) if i > 0 
                                                  else input_params['reserve'][0]) * 1e3, mutable=True, units=pyo.units.kW)
        blk.min_reserve = pyo.Constraint(expr=blk.total_reserve >= blk.reserve_over_1hr)

        # capacity requirement
        blk.cap_requirement = pyo.Constraint(expr=m.battery_system_capacity * capacity_credit_battery + m.turb_system_capacity >= capacity_requirement * 1e3)

        blk.hydrogen_revenue = pyo.Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * blk_tank.outlet_to_pipeline.flow_mol[0] * s_per_ts)
        blk.ng_costs = pyo.Expression(expr=blk.fs.ng_kg / mmbtu_to_ng_kg * input_params['NG_prices'][i])

        # calculate grid sales and purchases. Any sales over the max_sales is $0/MWh.
        if 'max_sales' in input_params.keys():
            blk.grid_sales_ub = pyo.Constraint(expr=blk.grid_sales - blk.grid_purchase <= input_params['max_sales'] * 1e3)
            blk.grid_sales.setub(input_params['max_sales'] * 1e3)
        if 'max_purchases' in input_params.keys():
            blk.grid_purchase_ub = pyo.Constraint(expr=blk.grid_purchase - blk.grid_sales <= input_params['max_purchases'] * 1e3)
            blk.grid_purchase.setub(input_params['max_purchases'] * 1e3)
        blk.grid_cost = pyo.Expression(expr=input_params['LMP'][i] * (blk.grid_purchase - blk.grid_sales) * 1e-3)
        blk.costs = pyo.Expression(expr=blk.grid_cost + blk.var_total_cost + blk.ng_costs)
        prev_turb_elec = turb_elec

    m.annual_revenue = pyo.Expression(expr=(sum([-blk.costs + blk.hydrogen_revenue for blk in blks])) * 52.143 / n_weeks
                                           - m.annual_fixed_cost)

    m.NPV = pyo.Expression(expr=-m.total_cap_cost + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV * 1e-3)


def pv_battery_hydrogen_optimize(n_time_points, input_params, verbose=False, plot=False):
    """
    The main function for optimizing the flowsheet's design and operating variables for Net Present Value. 

    Creates the MultiPeriodModel and adds the size and operating constraints in addition to the Net Present Value Objective.
    The NPV is a function of the capital costs, the electricity market profit, the hydrogen market profit, and the capital recovery factor.
    The operating decisions and state evolution of the unit models and the flowsheet as a whole form the constraints of the Non-linear Program.

    Required input parameters include:
        `pv_mw`: initial guess of the pv size
        `pv_mw_ub`: upper bound of pv size
        `batt_mw`: initial guess of the battery size
        `pem_mw`: initial guess of the pem size
        `pem_bar`: operating pressure
        `pem_temp`: operating temperature [K]
        `tank_size`: initial guess of the tank_size [kg H2]
        `turb_mw`: intial guess of the turbine size
        `ng_turb_conv`: ng conversion rate kWh/kgH2
        `h2_turb_conv`: h2 conversion rate kWh/kgH2
        `pv_resource`: dictionary of pv resource configs for each time point
        `h2_price_per_kg`: market price of hydrogen
        `LMP`: LMPs for each time point

    Args:
        n_time_points: number of periods in MultiPeriod model
        input_params: 
        verbose: print all logging and outputs from unit models, initialization, solvers, etc
        plot: plot the operating variables time series
    """
    # create the multiperiod model object
    n_weeks = n_time_points / (7 * 24)
    mp_model = MultiPeriodModel(n_time_points=n_time_points,
                                process_model_func=partial(pv_battery_hydrogen_mp_block, input_params=input_params, verbose=verbose),
                                linking_variable_func=pv_battery_hydrogen_variable_pairs,
                                periodic_variable_func=pv_battery_hydrogen_periodic_variable_pairs)

    mp_model.build_multi_period_model(input_params['pv_resource'])

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()
    # blks[0].fs.battery.initial_energy_throughput.fix()
    
    size_constraints(mp_model, input_params)
    
    # Add hydrogen market
    m.h2_price_per_kg = pyo.Param(default=input_params['h2_price_per_kg'], mutable=True)

    calculate_capital_costs(m, input_params)
    calculate_fixed_costs(m, input_params)
    calculate_variable_costs(mp_model, input_params)

    solvers_list = ['xpress_direct', 'cbc', 'ipopt']
    add_load_following_obj(mp_model, input_params)

    opt = None
    for solver in solvers_list:
        if pyo.SolverFactory(solver).available(exception_flag=False):
            opt = pyo.SolverFactory(solver)
            break
    if not opt:
        raise RuntimeWarning("No available solvers")

    if solver == 'ipopt':
        opt.options['tol'] = 1e-6
        opt.options['OF_ma27_liw_init_factor'] = 50
        opt.options['OF_ma27_la_init_factor'] = 50
        opt.options['OF_ma27_meminc_factor'] = 5
        # opt.options['max_iter'] = 200
        if "tempfile" in input_params.keys():
            opt.options['output_file'] = input_params['tempfile']

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)

    res = opt.solve(m, tee=True)

    if res.Solver.status != "ok":
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=False, log_variables=False)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
        raise RuntimeError("Did not solve")

    h2_prod = [pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * s_per_ts / h2_mols_per_kg) for i in range(n_time_points)]
    h2_tank_in = [pyo.value(blks[i].fs.h2_tank.inlet.flow_mol[0] * s_per_ts / h2_mols_per_kg) for i in range(n_time_points)]
    h2_tank_out = [pyo.value((blks[i].fs.h2_tank.outlet_to_pipeline.flow_mol[0] + blks[i].fs.h2_tank.outlet_to_turbine.flow_mol[0]) * s_per_ts / h2_mols_per_kg) for i in range(n_time_points)]
    h2_tank_holdup = [pyo.value(blks[i].fs.h2_tank.tank_holdup[0]) / h2_mols_per_kg for i in range(n_time_points)]
    h2_sales = [pyo.value(blks[i].fs.h2_tank.outlet_to_pipeline.flow_mol[0] * s_per_ts / h2_mols_per_kg) for i in range(n_time_points)]
    h2_turbine_in = [pyo.value(blks[i].fs.h2_kg) for i in range(n_time_points)]
    ng_turbine_in = [pyo.value(blks[i].fs.ng_kg) for i in range(n_time_points)]

    pv_gen = [pyo.value(blks[i].fs.pv.electricity[0]) for i in range(n_time_points)]
    pv_out = [pyo.value(blks[i].fs.splitter.grid_elec[0]) for i in range(n_time_points)]
    pv_to_pem = [pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)]
    batt_out = [pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)]
    batt_in = [pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)]
    batt_kwh = [pyo.value(blks[i].fs.battery.state_of_charge[0]) for i in range(n_time_points)]
    batt_soc = np.array(batt_kwh) / value(m.battery_system_energy)
    turbine_ng_elec = [pyo.value(blks[i].fs.turbine_ng_elec) for i in range(n_time_points)]
    turbine_h2_elec = [pyo.value(blks[i].fs.turbine_h2_elec) for i in range(n_time_points)]
    turbine_elec_total = [pyo.value(blks[i].fs.turbine_elec_total) for i in range(n_time_points)]
    grid_purchase = [pyo.value(blks[i].grid_purchase) for i in range(n_time_points)]
    grid_sales = [pyo.value(blks[i].grid_sales) for i in range(n_time_points)]
    
    excess_pv = [pyo.value(blks[i].excess_pv) for i in range(n_time_points)]
    battery_reserve = [pyo.value(blks[i].battery_reserve) for i in range(n_time_points)]
    turbine_reserve = [pyo.value(blks[i].turbine_reserve) for i in range(n_time_points)]
    pem_reserve = [pyo.value(blks[i].pem_reserve) for i in range(n_time_points)]
    total_reserve = [pyo.value(blks[i].total_reserve) for i in range(n_time_points)]
    min_reserve = [pyo.value(blks[i].reserve_over_1hr) for i in range(n_time_points)]

    ng_fuel = [pyo.value(blks[i].fs.ng_kg / mmbtu_to_ng_kg) for i in range(n_time_points)]
    ng_costs = [pyo.value(blks[i].ng_costs) for i in range(n_time_points)]
    grid_costs = [pyo.value(blks[i].grid_cost) for i in range(n_time_points)]
    pem_var_costs = [pyo.value(blks[i].fs.pem.var_cost) for i in range(n_time_points)]
    op_costs = [pyo.value(blks[i].var_total_cost) for i in range(n_time_points)]
    total_costs = [pyo.value(blks[i].costs) for i in range(n_time_points)]
    h2_revenue = [pyo.value(blks[i].hydrogen_revenue) for i in range(n_time_points)]

    marginal_cost_ng = [(input_params['NG_prices'][i]*ng_turbine_in[i]/mmbtu_to_ng_kg)/turbine_ng_elec[i] * 1e3 for i in range(n_time_points)]
    
    # calc hydrogen LCOE $/MWh
    discounted_energy = sum(sum(turbine_h2_elec) * 1e-3 / (1 + discount_rate) ** i for i in range(N))
    pem_cap_cost = pyo.value(m.pem_cap_cost * m.pem_system_capacity)
    pem_om_cost = pyo.value(m.pem_system_capacity * m.pem_op_cost_unit) + sum(pem_var_costs) 
    discounted_costs = pem_cap_cost + sum(pem_om_cost / (1 + discount_rate) ** i for i in range(n_time_points))
    discounted_costs_w_revenue = pem_cap_cost + sum((pem_om_cost - sum(h2_revenue)) / (1 + discount_rate) ** i for i in range(n_time_points))
    
    if discounted_energy != 0:
        h2_lcoe = discounted_costs / discounted_energy
        h2_lcoe_w_revenue = discounted_costs_w_revenue / discounted_energy
    else:
        h2_lcoe = 0
        h2_lcoe_w_revenue = 0

    hours = np.arange(n_time_points)

    pv_cap = value(m.pv_system_capacity + m.pv_add_system_capacity) * 1e-3
    batt_cap = value(m.battery_system_capacity) * 1e-3
    batt_energy = value(m.battery_system_energy) * 1e-3

    pem_cap = value(m.pem_system_capacity) * 1e-3
    tank_size = value(m.h2_tank_size) * kg_to_tons # to ton
    turb_cap = value(m.turb_system_capacity) * 1e-3

    co2_emissions = sum(ng_fuel) * co2_emissions_lb_mmbtu

    design_res = {
        'pv_mw': pv_cap,
        "batt_mw": batt_cap,
        "batt_mwh": batt_energy,
        "batt_hr": batt_energy / batt_cap if batt_cap else 0,
        "pem_mw": pem_cap,
        "tank_tonH2": tank_size,
        "turb_mw": turb_cap,
        "capital_cost": value(m.total_cap_cost),
        "capital_cost_pv": value(m.pv_cap_cost * m.pv_add_system_capacity),
        "capital_cost_batt_kw": value(m.batt_cap_cost_kw * m.battery_system_capacity),
        "capital_cost_batt_kwh": value(m.batt_cap_cost_kwh * m.battery_system_energy),
        "capital_cost_pem": value(m.pem_cap_cost * m.pem_system_capacity),
        "capital_cost_tank": value(m.tank_cap_cost * m.h2_tank_size),
        "capital_cost_turb": value(m.turb_cap_cost * (m.turb_system_capacity - input_params['turb_mw'] * 1e3)),
        "annual_costs_fixed": value(m.annual_fixed_cost),
        "fixed_cost_pv": value((m.pv_system_capacity + m.pv_add_system_capacity) * m.pv_op_cost_unit),
        "fixed_cost_pem": value(m.pem_system_capacity * m.pem_op_cost_unit),
        "fixed_cost_tank": value(m.h2_tank_size * m.h2_tank_op_cost_unit),
        "fixed_cost_turb": value(m.turb_system_capacity * m.h2_turbine_op_cost_unit),
        "annual_costs_variable": sum(op_costs),
        "variable_cost_batt": sum([pyo.value(blks[i].fs.battery.var_cost) for i in range(n_time_points)]),
        "variable_cost_pem": sum(pem_var_costs),
        "variable_cost_turb": sum([pyo.value(blks[i].turb_var_cost) for i in range(n_time_points)]),
        "annual_costs_NG": sum(ng_costs) * 52 / n_weeks,
        "annual_costs_grid": sum(grid_costs) * 52 / n_weeks,
        "annual_costs_total": sum(total_costs) * 52 / n_weeks,
        "annual_rev_h2": sum(h2_revenue) * 52 / n_weeks,
        "NPV": value(m.NPV),
        "H2_LCOE": h2_lcoe,
        "H2_LCOE_w_revenue": h2_lcoe_w_revenue,
        "NG_marginal_cost_avg": np.mean(np.array(marginal_cost_ng)[~np.isnan(marginal_cost_ng)]),
        "CO2_lb": co2_emissions
    }

    print(design_res)

    df = pd.DataFrame(index=range(n_time_points))
    df['Total PV Generation [MW]'] = np.array(pv_gen) * 1e-3
    df['Total Power Output [MW]'] = np.sum((pv_out, batt_out, turbine_elec_total), axis=0) * 1e-3
    df['PV Power Output [MW]'] = np.array(pv_out) * 1e-3
    df['PV Power to Battery [MW]'] = np.array(batt_in) * 1e-3
    df['State of Charge'] = np.array(batt_soc)
    df['Battery Power Output [MW]'] = np.array(batt_out) * 1e-3
    df['PV Power to PEM [MW]'] = np.array(pv_to_pem) * 1e-3
    df['PEM H2 Output [kg]'] = np.array(h2_prod)
    df['Tank H2 Input [kg]'] = np.array(h2_tank_in)
    df['H2 Sales [kg]'] = np.array(h2_sales)
    df['Turbine H2 Input [kg]'] = np.array(h2_turbine_in)
    df['Turbine NG Input [kg]'] = np.array(ng_turbine_in)
    df['Turbine Power [MW]'] = np.array(turbine_elec_total) * 1e-3
    df['Turbine Power from H2 [MW]'] = np.array(turbine_h2_elec) * 1e-3
    df['Turbine Power from NG [MW]'] = np.array(turbine_ng_elec) * 1e-3
    df['Purchased Power [MW]'] = np.array(grid_purchase) * 1e-3
    df['Sold Power [MW]'] = np.array(grid_sales) * 1e-3
    df['Tank Holdup [kg]'] = np.array(h2_tank_holdup)
    df['Excess PV [MW]'] = np.array(excess_pv) * 1e-3
    df['Battery Reserve [MW]'] = np.array(battery_reserve) * 1e-3
    df['PEM Reserve [MW]'] = np.array(pem_reserve) * 1e-3
    df['Turbine Reserve [MW]'] = np.array(turbine_reserve) * 1e-3
    df['Total Reserve [MW]'] = np.array(total_reserve) * 1e-3
    df['Min Reserve [MW]'] = np.array(min_reserve) * 1e-3
    df['Load [MW]'] = [pyo.value(blks[i].load_power) * 1e-3 for i in range(n_time_points)]

    df['Natural Gas [MMBTU]'] = np.array(ng_fuel)
    df['Natural Gas Cost [$]'] = np.array(ng_costs)
    df['Natural Gas Portion Marginal Cost [$]'] = np.array(marginal_cost_ng)
    df['Operating Cost [$]'] = np.array(op_costs)
    df['Total Cost [$]'] = np.array(total_costs)
    df['H2 Revenue [$]'] = np.array(h2_revenue)
    df['Grid Income [$]'] = np.array(grid_costs)

    return design_res, df


if __name__ == "__main__":
    max_sales = 1000
    max_purchases = 1000
    re_h2_parameters['max_sales'] = max_sales      # MW
    re_h2_parameters['max_purchases'] = max_purchases
    des_res, df_res = pv_battery_hydrogen_optimize(
        n_time_points=len(re_h2_parameters['pv_resource']),
        input_params=re_h2_parameters, verbose=False, plot=False)

    print(des_res)