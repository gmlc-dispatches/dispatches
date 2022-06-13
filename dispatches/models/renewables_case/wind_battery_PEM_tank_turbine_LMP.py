#################################################################################
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
#################################################################################
import numpy as np
import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.load_parameters import *

design_opt = True
extant_wind = True

pyo_model = None


def wind_battery_pem_tank_turb_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    if "Simple" in type(m1.fs.h2_tank).__name__:
        pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    else:
        pairs = [(m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')],
                m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')]),
                (m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap'])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
              (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput)]
    if design_opt:
        pairs += [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
        if "Simple" not in type(m1.fs.h2_tank).__name__:
            pairs += [(m1.fs.h2_tank.tank_length[0], m2.fs.h2_tank.tank_length[0])]
    return pairs


def wind_battery_pem_tank_turb_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    if "Simple" in type(m1.fs.h2_tank).__name__:
        pairs = [(m1.fs.h2_tank.tank_holdup[0], m2.fs.h2_tank.tank_holdup_previous[0])]
    else:
        pairs = [(m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')],
                m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')]),
                (m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap'])]
    pairs += [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge)]
    if design_opt:
        pairs += [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
        if "Simple" not in type(m1.fs.h2_tank).__name__:
            pairs += [(m1.fs.h2_tank.tank_length[0], m2.fs.h2_tank.tank_length[0])]
    return pairs


def wind_battery_pem_tank_turb_om_costs(m):
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


def initialize_mp(m, verbose=False):
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

    if not "Simple" in type(m.fs.h2_tank).__name__:
        m.fs.h2_tank.outlet.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
        m.fs.h2_tank.initialize(outlvl=outlvl)
        m.fs.h2_tank.outlet.flow_mol[0].unfix()
        if use_simple_h2_tank:
            m.fs.h2_tank.energy_balances.deactivate()
        if verbose:
            m.fs.h2_tank.report(dof=True)
    else:
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].fix()
        m.fs.h2_tank.tank_holdup_previous.fix(0)
        m.fs.h2_tank.initialize(outlvl=outlvl)
        m.fs.h2_tank.outlet_to_turbine.flow_mol[0].unfix()
        m.fs.h2_tank.outlet_to_pipeline.flow_mol[0].unfix()
        m.fs.h2_tank.tank_holdup_previous.unfix()

    if not "Simple" in type(m.fs.h2_tank).__name__:
        if hasattr(m.fs, "tank_valve"):
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

        propagate_state(m.fs.h2_splitter_to_sold)
        m.fs.tank_sold.initialize(outlvl=outlvl)
        if verbose:
            m.fs.tank_sold.report()

        propagate_state(m.fs.h2_splitter_to_turb)
    else:
        propagate_state(m.fs.h2_tank_to_turb)

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


def wind_battery_pem_tank_turb_model(wind_resource_config, tank_type, verbose):
    m = create_model(fixed_wind_mw, pem_bar, fixed_batt_mw, tank_type, fixed_tank_size, h2_turb_bar,
                     wind_resource_config, verbose)

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)

    if tank_type == "detailed":
        m.fs.h2_tank.previous_state[0].temperature.fix(PEM_temp)
        m.fs.h2_tank.previous_state[0].pressure.fix(pem_bar * 1e5)

    initialize_mp(m, verbose=verbose)

    if tank_type == "detailed":
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

    if design_opt:
        if not extant_wind:
            m.fs.windpower.system_capacity.unfix()
        if tank_type == "detailed":
            m.fs.h2_tank.tank_length.unfix()
        m.fs.battery.nameplate_power.unfix()
    return m


def wind_battery_pem_tank_turb_mp_block(wind_resource_config, tank_type, verbose):
    global pyo_model
    if pyo_model is None:
        pyo_model = wind_battery_pem_tank_turb_model(wind_resource_config, tank_type, verbose)
    m = pyo_model.clone()
    m.fs.windpower.config.resource_probability_density = wind_resource_config['resource_probability_density']
    m.fs.windpower.setup_resource()

    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
    m.fs.windpower.initialize(outlvl=outlvl)
    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.initialize()
    return m


def wind_battery_pem_tank_turb_optimize(n_time_points, h2_price=h2_price_per_kg, tank_type="simple", verbose=False, plot=False):
    from timeit import default_timer
    start = default_timer()
    # create the multiperiod model object
    mp_model = MultiPeriodModel(n_time_points=n_time_points,
                                process_model_func=partial(wind_battery_pem_tank_turb_mp_block, tank_type=tank_type, verbose=verbose),
                                linking_variable_func=wind_battery_pem_tank_turb_variable_pairs,
                                periodic_variable_func=wind_battery_pem_tank_turb_periodic_variable_pairs)

    mp_model.build_multi_period_model(wind_resource)

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    using_simple_tank = "Simple" in type(blks[0].fs.h2_tank).__name__
    if not using_simple_tank and use_simple_h2_tank:
        # turn off energy holdup constraints
        for blk in blks:
            if hasattr(blk, "link_constraints"):
                blk.link_constraints[1].deactivate()
            if hasattr(blk, "periodic_constraints"):
                blk.periodic_constraints[1].deactivate()

    m.h2_price_per_kg = pyo.Param(default=h2_price, mutable=True)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=fixed_pem_mw * 1e3, units=pyunits.kW)
    m.h2_tank_size = Var(domain=NonNegativeReals, initialize=fixed_tank_size)
    m.turb_system_capacity = Var(domain=NonNegativeReals, initialize=turb_p_mw * 1e3, units=pyunits.kW)
    if not design_opt:
        m.pem_system_capacity.fix(fixed_pem_mw * 1e3)
        m.h2_tank_size.fix(fixed_tank_size)
        m.turb_system_capacity.fix(turb_p_mw * 1e3)
        m.contract_capacity = Var(domain=NonNegativeReals, initialize=0, units=pyunits.mol / pyunits.second)

    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if extant_wind:
        m.wind_cap_cost.set_value(0.)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)
    m.tank_cap_cost = pyo.Param(default=tank_cap_cost_per_kg, mutable=True)
    m.turb_cap_cost = pyo.Param(default=turbine_cap_cost, mutable=True)

    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem
        blk_tank = blk.fs.h2_tank
        blk_turb = blk.fs.h2_turbine
        
        # add operating constraints
        blk_turb.electricity = Expression(blk_turb.flowsheet().config.time,
                                          rule=lambda b, t: (-b.turbine.work_mechanical[t]
                                                             - b.compressor.work_mechanical[t]) * 1e-3)

        # add operating costs
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
        if using_simple_tank:
            blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * (
                blk_tank.outlet_to_pipeline.flow_mol[0] - blk.fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600)
        else:
            blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg / h2_mols_per_kg * (
                blk.fs.tank_sold.flow_mol[0] - blk.fs.mixer.purchased_hydrogen_feed_state[0].flow_mol) * 3600)

    # add size constraints
    m.pem_max_p = Constraint(mp_model.pyomo_model.TIME,
                                rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)
    if using_simple_tank:
        m.tank_max_p = Constraint(mp_model.pyomo_model.TIME,
                                    rule=lambda b, t: blks[t].fs.h2_tank.tank_holdup[0] <= m.h2_tank_size)
    else:
        m.tank_max_p = Constraint(mp_model.pyomo_model.TIME,
                                    rule=lambda b, t: blks[t].fs.h2_tank.material_holdup[0, "Vap", "hydrogen"] <= m.h2_tank_size)
    m.turb_max_p = Constraint(mp_model.pyomo_model.TIME,
                              rule=lambda b, t: blks[t].fs.h2_turbine.electricity[0] <= m.turb_system_capacity)

    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(prices_used[i] * 1e-3)     # to $/kWh

    n_weeks = n_time_points / (7 * 24)

    m.annual_revenue = Expression(expr=(sum([blk.profit + blk.hydrogen_revenue for blk in blks])) * 52.143 / n_weeks)

    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity
                              + m.batt_cap_cost * blks[0].fs.battery.nameplate_power
                              + m.pem_cap_cost * m.pem_system_capacity
                              + m.tank_cap_cost * m.h2_tank_size
                              + m.turb_cap_cost * m.turb_system_capacity
                              ) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV * 1e-8)

    blks[0].fs.windpower.system_capacity.setub(wind_ub_mw * 1e3)
    # blks[0].fs.battery.initial_state_of_charge.fix(0)
    blks[0].fs.battery.initial_energy_throughput.fix(0)

    opt = pyo.SolverFactory('ipopt')

    opt.options['max_iter'] = 100000
    opt.options['tol'] = 1e-6

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
    if using_simple_tank:
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
    lmp_array = weekly_prices[0:n_time_points].flatten()
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

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3
    batt_cap = value(blks[0].fs.battery.nameplate_power) * 1e-3
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
        fig, ax1 = plt.subplots(3, 1, figsize=(12, 8))
        plt.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(batt_cap, 2)} MW Battery, "
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

    plt.show()

    return design_res


if __name__ == "__main__":
    wind_battery_pem_tank_turb_optimize(n_time_points=7 * 24, h2_price=h2_price_per_kg, verbose=False, plot=True)