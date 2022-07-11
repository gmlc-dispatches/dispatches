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
import numpy as np
import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *


def wind_battery_pem_tank_turb_variable_pairs(m1, m2, tank_type):
    """
    This function links together unit model state variables from one timestep to the next.

    The hydrogen tank and the battery model have material and energy holdups that need to be consistent across time blocks.
    If using the `simple` tank model, there are no energy holdups to account for. For the `detailed` tank model, the emergy
    holdups need to be linked.

    Args:
        m1: current time block model
        m2: next time block model
        tank_type: `simple`, `detailed` or `detailed-valve`
    """
    if tank_type == "simple":
        pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    else:
        pairs = [(m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')],
                m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')])]
        if tank_type == 'detailed-valve':
            pairs += [(m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap'])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput),
              (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    
    return pairs


def wind_battery_pem_tank_turb_periodic_variable_pairs(m1, m2, tank_type):
    """
    The final hydrogen material holdup and battery storage of charge must be the same as in the intial timestep. 
    If using the `simple` tank model, there are no energy holdups to account for. For the `detailed` tank model, the emergy
    holdups need to be linked.

    Args:
        m1: final time block model
        m2: first time block model
        tank_type: `simple`, `detailed` or `detailed-valve`
    """
    if tank_type == "simple":
        pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    else:
        pairs = [(m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')],
                m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')])]
        if tank_type == 'detailed-valve':
            pairs += [(m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap'])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    return pairs


def wind_battery_pem_tank_turb_om_costs(m):
    """
    Add unit fixed and variable operating costs as parameters for the unit model m
    """
    m.fs.windpower.op_cost = pyo.Param(
        initialize=wind_op_cost,
        doc="fixed cost of operating wind plant $/kW-yr")
    m.fs.pem.op_cost = pyo.Param(
        initialize=pem_op_cost,
        doc="fixed cost of operating pem $/kW-yr"
    )
    m.fs.pem.var_cost = pyo.Param(
        initialize=pem_var_cost,
        doc="variable operating cost of pem $/kWh"
    )
    m.fs.h2_tank.op_cost = Expression(
        expr=tank_op_cost,
        doc="fixed cost of operating tank in $/m^3"
    )
    m.fs.h2_turbine.op_cost = Expression(
        expr=turbine_op_cost,
        doc="fixed cost of operating turbine $/kW-pr"
    )
    m.fs.h2_turbine.var_cost = Expression(
        expr=turbine_var_cost,
        doc="variable operating cost of turbine $/kg"
    )


def initialize_fs(m, tank_type, verbose=False):
    """
    Initializing the flowsheet is done starting with the wind model and propagating the solved initial state to downstream models.

    The splitter is initialized with no flow to the battery or PEM so all electricity flows to the grid, which makes the initialization of all
    unit models downstream of the wind plant independent of its time-varying electricity production. This initialzation function can
    then be repeated for all timesteps within a dynamic analysis.

    Args:
        m: model
        tank_type: `simple`, `detailed` or `detailed-valve`
        verbose:
    """
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING

    m.fs.windpower.initialize(outlvl=outlvl)

    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.battery_elec[0].fix(0)
    m.fs.splitter.pem_elec[0].fix(0)
    m.fs.splitter.initialize()
    m.fs.splitter.battery_elec[0].unfix()
    m.fs.splitter.pem_elec[0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_pem)
    propagate_state(m.fs.splitter_to_battery)

    m.fs.battery.elec_in[0].fix()
    m.fs.battery.elec_out[0].fix(value(m.fs.battery.elec_in[0]))
    m.fs.battery.initialize(outlvl=outlvl)
    m.fs.battery.elec_in[0].unfix()
    m.fs.battery.elec_out[0].unfix()
    if verbose:
        m.fs.battery.report(dof=True)

    m.fs.pem.initialize(outlvl=outlvl)
    if verbose:
        m.fs.pem.report(dof=True)

    propagate_state(m.fs.pem_to_tank)

    if tank_type == "simple":
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix(0)
        m.fs.h2_tank.tank_holdup_previous.fix(0)
        m.fs.h2_tank.initialize(outlvl=outlvl)
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].unfix()
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].unfix()
        m.fs.h2_tank.tank_holdup_previous.unfix()
    else:
        m.fs.h2_tank.outlet.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
        m.fs.h2_tank.initialize(outlvl=outlvl)
        m.fs.h2_tank.outlet.flow_mol[0].unfix()
        if verbose:
            m.fs.h2_tank.report(dof=True)

    if tank_type == "simple":
        propagate_state(m.fs.h2_tank_to_turb)
    else:
        if tank_type == "detailed-valve":
            propagate_state(m.fs.tank_to_valve)
            m.fs.tank_valve.initialize(outlvl=outlvl)
            if verbose:
                m.fs.tank_valve.report(dof=True)

        propagate_state(m.fs.valve_to_h2_splitter)
        m.fs.h2_splitter.split_fraction[0, "sold"].fix(0.5)
        m.fs.h2_splitter.initialize(outlvl=outlvl)
        m.fs.h2_splitter.split_fraction[0, "sold"].unfix()
        if verbose:
            m.fs.h2_splitter.report(dof=True)

        propagate_state(m.fs.h2_splitter_to_turb)

    m.fs.translator.initialize(outlvl=outlvl)
    if verbose:
        m.fs.translator.report(dof=True)

    propagate_state(m.fs.translator_to_mixer)
    m.fs.mixer.air_h2_ratio.deactivate()
    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].fix(h2_turb_min_flow)
    h2_out = value(m.fs.mixer.hydrogen_feed.flow_mol[0] + m.fs.mixer.purchased_hydrogen_feed.flow_mol[0])
    m.fs.mixer.air_feed.flow_mol[0].fix(h2_out * air_h2_ratio)
    m.fs.mixer.initialize(outlvl=outlvl)
    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].unfix()
    m.fs.mixer.air_feed.flow_mol[0].unfix()
    m.fs.mixer.air_h2_ratio.activate()
    if verbose:
        m.fs.mixer.report(dof=True)

    propagate_state(m.fs.mixer_to_turbine)

    m.fs.h2_turbine.initialize(outlvl=outlvl)
    if verbose:
        m.fs.h2_turbine.report(dof=True)


def wind_battery_pem_tank_turb_model(wind_resource_config, input_params, verbose):
    """
    Creates an initialized flowsheet model for a single time step with operating, size and cost components
    
    First, the model is created using the input_params and wind_resource_config
    Second, the model is initialized so that it solves and its values are internally consistent
    Third, battery ramp constraints and operating cost components are added

    Args:
        wind_resource_config: wind resource for the time step
        input_params: size and operation parameters. Required keys: `wind_mw`, `pem_bar`, `batt_mw`, `tank_type`, `tank_size`, `pem_bar`
        verbose:
    """
    m = create_model(input_params['wind_mw'], input_params['pem_bar'], input_params['batt_mw'], input_params['tank_type'], input_params['tank_size'], input_params['pem_bar'],
                     wind_resource_config)

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)

    if input_params['tank_type'] == "detailed":
        m.fs.h2_tank.previous_state[0].temperature.fix(input_params['pem_temp'])
        m.fs.h2_tank.previous_state[0].pressure.fix(input_params['pem_bar'] * 1e5)

    initialize_fs(m, input_params['tank_type'], verbose=verbose)

    if input_params['tank_type'] == "detailed":
        m.fs.h2_tank.previous_state[0].temperature.unfix()
        m.fs.h2_tank.previous_state[0].pressure.unfix()
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()

    batt = m.fs.battery
    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge <= battery_ramp_rate)

    wind_battery_pem_tank_turb_om_costs(m)

    return m


def wind_battery_pem_tank_turb_mp_block(wind_resource_config, input_params, verbose):
    """
    Wrapper of `wind_battery_pem_tank_turb_model` for creating the process model per time point for the MultiPeriod model.
    Uses cloning of the Pyomo model in order to reduce runtime. 
    The clone is reinitialized with the `wind_resource_config` for the given time point, which only required modifying
    the windpower and the splitter, as the rest of the units have no flow and therefore is unaffected by wind resource changes.

    Args:
        wind_resource_config: dictionary with `resource_speed` for the time step
        input_params: size and operation parameters. Required keys: `wind_mw`, `pem_bar`, `batt_mw`, `tank_type`, `tank_size`, `pem_bar`
        verbose:
    """

    if 'pyo_model' not in input_params.keys():
        input_params['pyo_model'] = wind_battery_pem_tank_turb_model(wind_resource_config, input_params, verbose)
    m = input_params['pyo_model'].clone()

    if 'resource_speed' in wind_resource_config.keys():
        m.fs.windpower.config.resource_speed = wind_resource_config['resource_speed']
    elif 'capacity_factor' in wind_resource_config.keys():
        m.fs.windpower.config.capacity_factor = wind_resource_config['capacity_factor']
    else:
        raise ValueError(f"`wind_resource_config` dict must contain either 'resource_speed' or 'capacity_factor' values")

    m.fs.windpower.setup_resource()

    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
    m.fs.windpower.initialize(outlvl=outlvl)
    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.initialize()
    return m


def wind_battery_pem_tank_turb_optimize(n_time_points, input_params, verbose=False, plot=False):
    """
    The main function for optimizing the flowsheet's design and operating variables for Net Present Value. 

    Creates the MultiPeriodModel and adds the size and operating constraints in addition to the Net Present Value Objective.
    The NPV is a function of the capital costs, the electricity market profit, the hydrogen market profit, and the capital recovery factor.
    The operating decisions and state evolution of the unit models and the flowsheet as a whole form the constraints of the Non-linear Program.

    Required input parameters include:
        `wind_mw`: initial guess of the wind size
        `wind_mw_ub`: upper bound of wind size
        `batt_mw`: initial guess of the battery size
        `pem_mw`: initial guess of the pem size
        `pem_bar`: operating pressure
        `pem_temp`: operating temperature [K]
        `tank_size`: initial guess of the tank_size [kg H2]
        `tank_type`: "simple", "detailed" or "detailed-valve"
        `turb_mw`: intial guess of the turbine size
        `wind_resource`: dictionary of wind resource configs for each time point
        `h2_price_per_kg`: market price of hydrogen
        `DA_LMPs`: LMPs for each time point
        `design_opt`: true to optimize design, else sizes are fixed at initial guess sizes
        `extant_wind`: if true, fix wind size to initial size and do not add wind capital cost to NPV

    Args:
        n_time_points: number of periods in MultiPeriod model
        input_params: 
        verbose: print all logging and outputs from unit models, initialization, solvers, etc
        plot: plot the operating variables time series
    """
    # create the multiperiod model object
    tank_type = input_params['tank_type']
    mp_model = MultiPeriodModel(n_time_points=n_time_points,
                                process_model_func=partial(wind_battery_pem_tank_turb_mp_block, input_params=input_params, verbose=verbose),
                                linking_variable_func=partial(wind_battery_pem_tank_turb_variable_pairs, tank_type=tank_type),
                                periodic_variable_func=partial(wind_battery_pem_tank_turb_periodic_variable_pairs, tank_type=tank_type))

    mp_model.build_multi_period_model(input_params['wind_resource'])

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()
    blks[0].fs.battery.initial_energy_throughput.fix(0)
    
    m.wind_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['wind_mw'] * 1e3, units=pyunits.kW, bounds=(0, input_params['wind_mw_ub'] * 1e3))
    m.battery_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['batt_mw'] * 1e3, units=pyunits.kW)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['pem_mw'] * 1e3, units=pyunits.kW)
    m.h2_tank_size = Var(domain=NonNegativeReals, initialize=input_params['tank_size'])
    m.turb_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['turb_mw'] * 1e3, units=pyunits.kW)

    # add size constraints
    if input_params['design_opt']:
        if input_params['extant_wind']:
            m.wind_system_capacity.fix()
        for blk in blks:
            if tank_type == "detailed":
                blk.fs.h2_tank.tank_length.unfix()
            blk.fs.battery.nameplate_power.unfix()
    else:
        m.pem_system_capacity.fix(input_params['pem_mw'])
        m.h2_tank_size.fix(input_params['tank_size'])
        m.turb_system_capacity.fix(input_params['turb_mw'])

    m.wind_max_p = Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.windpower.system_capacity <= m.wind_system_capacity)
    m.battery_max_p = Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_power <= m.battery_system_capacity)
    m.pem_max_p = Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)
    if tank_type == "simple":
        m.tank_max_p = Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.h2_tank.tank_holdup[0] <= m.h2_tank_size)
    else:
        m.tank_max_p = Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.h2_tank.material_holdup[0, "Vap", "hydrogen"] <= m.h2_tank_size)
    m.turb_max_p = Constraint(mp_model.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.h2_turbine.electricity[0] <= m.turb_system_capacity)

    # Add cost and price equations
    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if input_params['extant_wind']:
        m.wind_cap_cost.set_value(0.)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)
    m.tank_cap_cost = pyo.Param(default=tank_cap_cost_per_kg, mutable=True)
    m.turb_cap_cost = pyo.Param(default=turbine_cap_cost, mutable=True)

    m.contract_capacity = Var(domain=NonNegativeReals, initialize=0, units=pyunits.mol / pyunits.second)
    m.h2_price_per_kg = pyo.Param(default=input_params['h2_price_per_kg'], mutable=True)

    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem
        blk_tank = blk.fs.h2_tank
        blk_turb = blk.fs.h2_turbine

        # calculate operating costs
        blk_wind.op_total_cost = Expression(
            expr=blk_wind.system_capacity * blk_wind.op_cost / 8760,
        )
        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
        )
        blk_tank.op_total_cost = Expression(
            expr=m.h2_tank_size * blk_tank.op_cost / 8760
        )
        blk_turb.op_total_cost = Expression(
            expr=m.turb_system_capacity * blk_turb.op_cost / 8760 + blk_turb.var_cost * blk_turb.electricity[0]
        )

        # add market data for each block
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal * (blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0] + blk_turb.electricity[0])
        blk.profit = pyo.Expression(expr=blk.revenue
                                         - blk_wind.op_total_cost
                                         - blk_pem.op_total_cost
                                         - blk_tank.op_total_cost
                                         - blk_turb.op_total_cost
                                    )
        if tank_type == "simple":
            blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * (
                blk_tank.outlet_to_pipeline.flow_mol[0] - blk.fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600)
        else:
            blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * (
                blk.fs.h2_splitter.sold.flow_mol[0] - blk.fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600)

    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(input_params['DA_LMPs'][i] * 1e-3)     # to $/kWh

    n_weeks = n_time_points / (7 * 24)

    m.annual_revenue = Expression(expr=(sum([blk.profit + blk.hydrogen_revenue for blk in blks])) * 52.143 / n_weeks)

    m.NPV = Expression(expr=-(m.wind_cap_cost * m.wind_system_capacity
                              + m.batt_cap_cost * m.battery_system_capacity
                              + m.pem_cap_cost * m.pem_system_capacity
                              + m.tank_cap_cost * m.h2_tank_size
                              + m.turb_cap_cost * m.turb_system_capacity
                              ) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV * 1e-8)


    opt = pyo.SolverFactory('ipopt')

    opt.options['max_iter'] = 10000

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)

    opt.solve(m, tee=verbose)

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=False, log_variables=False)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)


    h2_prod = []
    wind_to_grid = []
    wind_to_pem = []
    wind_to_batt = []
    batt_to_grid = []
    soc = []
    wind_gen = []
    h2_tank_in = []
    h2_tank_out = []
    h2_tank_holdup = []
    h2_purchased = []
    h2_turbine_elec = []
    comp_kwh = []
    turb_kwh = []
    h2_revenue = []
    elec_income = []

    h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600 / 500) for i in range(n_time_points)])
    h2_tank_in.append([pyo.value(blks[i].fs.h2_tank.inlet.flow_mol[0] * 3600 / 500) for i in range(n_time_points)])
    if tank_type == "simple":
        h2_tank_out.append([pyo.value((blks[i].fs.h2_tank.outlet_to_pipeline.flow_mol[0] + blks[i].fs.h2_tank.outlet_to_turbine.flow_mol[0]) * 3600 / 500) for i in range(n_time_points)])
        h2_tank_holdup.append([pyo.value(blks[i].fs.h2_tank.tank_holdup[0]) for i in range(n_time_points)])
    else:  
        h2_tank_out.append([pyo.value(blks[i].fs.h2_tank.outlet.flow_mol[0] * 3600 / 500) for i in range(n_time_points)])
        h2_tank_holdup.append(
            [pyo.value(blks[i].fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]) for i in range(n_time_points)])
    h2_purchased.append([pyo.value(blks[i].fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600 / h2_mols_per_kg for i in range(n_time_points)])

    wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])
    wind_to_grid.append([pyo.value(blks[i].fs.splitter.grid_elec[0]) for i in range(n_time_points)])
    wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)])
    batt_to_grid.append([pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)])
    wind_to_batt.append([pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)])
    h2_turbine_elec.append([pyo.value(blks[i].fs.h2_turbine.electricity[0]) for i in range(n_time_points)])
    turb_kwh.append(
        [pyo.value(blks[i].fs.h2_turbine.turbine.work_mechanical[0]) * -1e-3 for i in range(n_time_points)])
    comp_kwh.append(
        [pyo.value(blks[i].fs.h2_turbine.compressor.work_mechanical[0]) * 1e-3 for i in range(n_time_points)])
    elec_income.append([pyo.value(blks[i].profit) for i in range(n_time_points)])
    h2_revenue.append([pyo.value(blks[i].hydrogen_revenue) for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points)
    lmp_array = input_params['DA_LMPs'][0:n_time_points]
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()
    h2_tank_in = np.asarray(h2_tank_in[0:n_weeks_to_plot]).flatten()
    h2_tank_out = np.asarray(h2_tank_out[0:n_weeks_to_plot]).flatten()
    h2_tank_holdup = np.asarray(h2_tank_holdup[0:n_weeks_to_plot]).flatten()
    batt_out = np.asarray(batt_to_grid[0:n_weeks_to_plot]).flatten()
    batt_in = np.asarray(wind_to_batt[0:n_weeks_to_plot]).flatten()
    h2_purchased = np.asarray(h2_purchased[0:n_weeks_to_plot]).flatten()
    turb_kwh = np.asarray(turb_kwh[0:n_weeks_to_plot]).flatten()
    comp_kwh = np.asarray(comp_kwh[0:n_weeks_to_plot]).flatten()
    h2_turbine_elec = np.asarray(h2_turbine_elec[0:n_weeks_to_plot]).flatten()
    h2_revenue = np.asarray(h2_revenue[0:n_weeks_to_plot]).flatten()
    elec_income = np.asarray(elec_income[0:n_weeks_to_plot]).flatten()

    wind_cap = value(m.wind_system_capacity) * 1e-3
    batt_cap = value(m.battery_system_capacity) * 1e-3
    pem_cap = value(m.pem_system_capacity) * 1e-3
    tank_size = value(m.h2_tank_size)
    turb_cap = value(m.turb_system_capacity) * 1e-3

    design_res = {
        'wind_mw': wind_cap,
        "batt_mw": batt_cap,
        "pem_mw": pem_cap,
        "tank_kgH2": tank_size,
        "turb_mw": turb_cap,
        "avg_turb_eff": np.average(turb_kwh/comp_kwh),
        "annual_rev_h2": sum(h2_revenue) * 52 / n_weeks,
        "annual_rev_E": sum(elec_income) * 52 / n_weeks,
        "NPV": value(m.NPV)
    }

    print(design_res)

    if plot:
        fig, ax1 = plt.subplots(3, 1, figsize=(20, 8))
        fig.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(batt_cap, 2)} MW Battery, "
                     f"{round(pem_cap, 2)} MW PEM, {round(tank_size, 2)} kgH2 Tank and {round(turb_cap, 2)} MW Turbine")

        # color = 'tab:green'
        ax1[0].set_xlabel('Hour')
        # ax1[0].set_ylabel('kW', )
        ax1[0].step(hours, wind_gen, label="Wind Generation [kW]")
        ax1[0].step(hours, wind_out, label="Wind to Grid [kW]")
        ax1[0].step(hours, wind_to_pem, label="Wind to Pem [kW]")
        ax1[0].step(hours, batt_in, label="Wind to Batt [kW]")
        ax1[0].step(hours, batt_out, label="Batt to Grid [kW]")
        ax1[0].step(hours, h2_turbine_elec, label="H2 Turbine [kW]")
        ax1[0].tick_params(axis='y', )
        ax1[0].legend()
        ax1[0].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        ax1[0].minorticks_on()
        ax1[0].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)

        ax2 = ax1[0].twinx()
        color = 'k'
        ax2.set_ylabel('LMP [$/MWh]', color=color)
        ax2.plot(hours, lmp_array[0:len(hours)], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # ax1[1].set_xlabel('Hour')
        # ax1[1].set_ylabel('kg/hr', )
        ax1[1].step(hours, h2_prod, label="PEM H2 production [kg/hr]")
        ax1[1].step(hours, h2_tank_in, label="Tank inlet [kg/hr]")
        ax1[1].step(hours, h2_tank_out, label="Tank outlet [kg/hr]")
        ax1[1].step(hours, h2_tank_holdup, label="Tank holdup [kg]")
        ax1[1].step(hours, h2_purchased, label="H2 purchased [kg/hr]")
        ax1[1].tick_params(axis='y', )
        ax1[1].legend()
        ax1[1].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        ax1[1].minorticks_on()
        ax1[1].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)

        ax2 = ax1[1].twinx()
        color = 'k'
        ax2.set_ylabel('LMP [$/MWh]', color=color)
        ax2.plot(hours, lmp_array[0:len(hours)], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        ax1[2].set_xlabel('Hour')
        ax1[2].step(hours, elec_income, label="Elec Income")
        ax1[2].step(hours, h2_revenue, label="H2 rev")
        ax1[2].step(hours, np.cumsum(elec_income), label="Elec Income cumulative")
        ax1[2].step(hours, np.cumsum(h2_revenue), label="H2 rev cumulative")
        ax1[2].legend()
        ax1[2].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        ax1[2].minorticks_on()
        ax1[2].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)
        fig.tight_layout()

    plt.show()

    return design_res


if __name__ == "__main__":
    default_input_params['wind_mw'] = 200
    des_res = wind_battery_pem_tank_turb_optimize(n_time_points=6 * 24, input_params=default_input_params, verbose=False, plot=True)
    print(des_res)