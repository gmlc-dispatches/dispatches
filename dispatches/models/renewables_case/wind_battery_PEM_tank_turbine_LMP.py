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
import numpy as np
import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_parameters import *

design_opt = True
extant_wind = True
extant_turbine = False


def wind_battery_pem_tank_turb_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    pairs = [(m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap']),
             (m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')],
              m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')]),
             (m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
             (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput)]
    if design_opt:
        pairs += [(m1.fs.h2_tank.tank_length[0], m2.fs.h2_tank.tank_length[0]),
                  (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
    return pairs


def wind_battery_pem_tank_turb_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    pairs = [(m1.fs.h2_tank.energy_holdup[0, 'Vap'], m2.fs.h2_tank.previous_energy_holdup[0, 'Vap']),
             (m1.fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')],
              m2.fs.h2_tank.previous_material_holdup[0, ('Vap', 'hydrogen')]),
             (m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge)]
    if design_opt:
        pairs += [(m1.fs.h2_tank.tank_length[0], m2.fs.h2_tank.tank_length[0]),
                  (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
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
        doc="variable operating cost of turbine $/kWh"
    )


def initialize_mp(m, verbose=False):
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING

    m.fs.windpower.initialize(outlvl=outlvl)

    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.split_fraction['grid', 0].fix(0.5)
    m.fs.splitter.split_fraction['battery', 0].fix(0.0)
    m.fs.splitter.split_fraction['pem', 0].fix(0.5)
    m.fs.splitter.initialize()
    m.fs.splitter.split_fraction['grid', 0].unfix()
    m.fs.splitter.split_fraction['battery', 0].unfix()
    m.fs.splitter.split_fraction['pem', 0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_grid)
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

    m.fs.h2_tank.outlet.flow_mol[0].fix(value(m.fs.h2_tank.inlet.flow_mol[0]))
    m.fs.h2_tank.initialize(outlvl=outlvl)
    m.fs.h2_tank.outlet.flow_mol[0].unfix()
    if verbose:
        m.fs.h2_tank.report(dof=True)

    if hasattr(m.fs, "tank_valve"):
        propagate_state(m.fs.tank_to_valve)
        # m.fs.tank_valve.outlet.flow_mol[0].fix(value(m.fs.tank_valve.inlet.flow_mol[0]))
        m.fs.tank_valve.initialize(outlvl=outlvl)
        # m.fs.tank_valve.outlet.flow_mol[0].unfix()
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
    m.fs.translator.initialize(outlvl=outlvl)
    if verbose:
        m.fs.translator.report(dof=True)

    propagate_state(m.fs.translator_to_mixer)
    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].fix(h2_turb_min_flow)

    # initial guess of air feed that will be needed to balance out hydrogen feed
    h2_out = value(m.fs.h2_tank.outlet.flow_mol[0] + m.fs.mixer.purchased_hydrogen_feed.flow_mol[0])
    m.fs.mixer.air_feed.flow_mol[0].fix(h2_out * 8)
    m.fs.mixer.initialize(outlvl=outlvl)
    m.fs.mixer.air_feed.flow_mol[0].unfix()
    m.fs.mixer.purchased_hydrogen_feed.flow_mol[0].unfix()
    if verbose:
        m.fs.mixer.report(dof=True)

    propagate_state(m.fs.mixer_to_turbine)
    m.fs.h2_turbine.compressor.ratioP[0].fix(25)
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_comp['nitrogen'].unfix()
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_comp['water'].unfix()
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_comp['oxygen'].unfix()
    # m.fs.h2_turbine.compressor.control_volume.properties_in[0].mole_frac_comp['argon'].unfix()

    m.fs.h2_turbine.turbine.ratioP[0].fix(1/25)
    m.fs.h2_turbine.initialize(outlvl=outlvl)
    m.fs.h2_turbine.compressor.ratioP[0].unfix()
    m.fs.h2_turbine.turbine.ratioP[0].unfix()
    if verbose:
        m.fs.h2_turbine.report(dof=True)


def wind_battery_pem_tank_turb_model(wind_resource_config, verbose):
    m = create_model(fixed_wind_mw, pem_bar, fixed_batt_mw, valve_cv, fixed_tank_len_m, h2_turb_bar,
                     wind_resource_config, verbose)

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)

    m.fs.h2_tank.previous_state[0].temperature.fix(PEM_temp)
    m.fs.h2_tank.previous_state[0].pressure.fix(pem_bar * 1e5)
    if hasattr(m.fs, "tank_valve"):
        m.fs.tank_valve.outlet.pressure[0].fix(1e5)
    # print(degrees_of_freedom(m))
    initialize_mp(m, verbose=verbose)
    # print(degrees_of_freedom(m))
    m.fs.h2_tank.previous_state[0].temperature.unfix()
    m.fs.h2_tank.previous_state[0].pressure.unfix()
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()

    if verbose:
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                           tag="properties")
        log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
        # log_close_to_bounds(m, logger=solve_log)

    wind_battery_pem_tank_turb_om_costs(m)

    if design_opt:
        if not extant_wind:
            m.fs.windpower.system_capacity.unfix()
        m.fs.h2_tank.tank_length.unfix()
        m.fs.battery.nameplate_power.unfix()
    return m


def wind_battery_pem_tank_turb_mp_block(wind_resource_config, verbose):
    m = wind_battery_pem_tank_turb_model(wind_resource_config, verbose)
    batt = m.fs.battery

    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge <= battery_ramp_rate)

    return m


def wind_battery_pem_tank_turb_optimize(verbose=False):
    # create the multiperiod model object
    mp_model = MultiPeriodModel(n_time_points=n_time_points,
                                process_model_func=partial(wind_battery_pem_tank_turb_model, verbose=verbose),
                                linking_variable_func=wind_battery_pem_tank_turb_variable_pairs,
                                periodic_variable_func=wind_battery_pem_tank_turb_periodic_variable_pairs)

    mp_model.build_multi_period_model(wind_resource)

    m = mp_model.pyomo_model
    blks = mp_model.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=h2_price_per_kg, mutable=True)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=fixed_pem_mw, units=pyunits.kW)
    m.turb_system_capacity = Var(domain=NonNegativeReals, initialize=turb_p_lower_bound, units=pyunits.kW)
    if not design_opt:
        m.pem_system_capacity.fix(fixed_pem_mw)
        m.turb_system_capacity.fix(turb_p_lower_bound)
    if h2_contract:
        m.contract_capacity = Var(domain=NonNegativeReals, initialize=0, units=pyunits.mol / pyunits.second)

    m.h2_tank_volume = pyo.Expression(
        expr=(blks[0].fs.h2_tank.tank_diameter[0] / 2) ** 2 * np.pi * blks[0].fs.h2_tank.tank_length[0])

    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem
        blk_tank = blk.fs.h2_tank
        blk_turb = blk.fs.h2_turbine
        # add operating constraints
        blk_pem.max_p = Constraint(blk_pem.flowsheet().config.time,
                                   rule=lambda b, t: b.electricity[t] <= m.pem_system_capacity)
        blk_turb.electricity = Expression(blk_turb.flowsheet().config.time,
                                          rule=lambda b, t: -b.turbine.work_mechanical[0] * 1e-3)
        blk_turb.max_p = Constraint(blk_turb.flowsheet().config.time,
                                    rule=lambda b, t: b.electricity[t] <= m.turb_system_capacity)
        # blk_turb.min_f = Constraint(blk_turb.flowsheet().config.time,
        #                             rule=lambda b, t: b.compressor.control_volume.properties_in[0].flow_mol >= h2_turb_min_flow)
        blk_turb.compressor.control_volume.properties_in[0].flow_mol.setlb(h2_turb_min_flow)
        # add operating costs
        blk_wind.op_total_cost = Expression(
            expr=blk_wind.system_capacity * blk_wind.op_cost / 8760,
        )
        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
        )
        blk_tank.op_total_cost = Expression(
            expr=m.h2_tank_volume * blk_tank.op_cost / 8760
        )
        blk_turb.op_total_cost = Expression(
            expr=m.turb_system_capacity * blk_turb.op_cost / 8760 + blk_turb.var_cost * blk_turb.electricity[0]
        )

        # add market data for each block
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal * (blk.fs.wind_to_grid[0] + blk_battery.elec_out[0] + blk_turb.electricity[0]) * 1e-3
        blk.profit = pyo.Expression(expr=blk.revenue
                                         - blk_wind.op_total_cost
                                         - blk_pem.op_total_cost
                                         - blk_tank.op_total_cost
                                         - blk_turb.op_total_cost
                                    )
        if h2_contract:
            blk.tank_contract = Constraint(blk_pem.flowsheet().config.time,
                                          rule=lambda b, t: m.contract_capacity <= blk.fs.tank_sold.flow_mol[t])

    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if extant_wind:
        m.wind_cap_cost.set_value(0.)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)
    m.tank_cap_cost = pyo.Param(default=tank_cap_cost, mutable=True)
    m.turb_cap_cost = pyo.Param(default=turbine_cap_cost, mutable=True)
    if extant_turbine:
        m.turb_cap_cost.set_value(0.)

    n_weeks = 1
    if h2_contract:
        m.hydrogen_revenue = Expression(expr=m.h2_price_per_kg * m.contract_capacity / h2_mols_per_kg
                                             * 3600 * n_time_points)
    else:
        m.hydrogen_revenue = Expression(
            expr=sum([m.h2_price_per_kg * (
                    blk.fs.tank_sold.flow_mol[0] - blk.fs.mixer.purchased_hydrogen_feed.flow_mol[0])
                      / h2_mols_per_kg
                      * 3600 for blk in blks]))

    m.annual_revenue = Expression(expr=(sum([blk.profit for blk in blks]) + m.hydrogen_revenue) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity
                              + m.batt_cap_cost * blks[0].fs.battery.nameplate_power
                              + m.pem_cap_cost * m.pem_system_capacity
                              + m.tank_cap_cost * m.h2_tank_volume
                              + m.turb_cap_cost * m.turb_system_capacity
                              ) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)
    # m.obj = pyo.Objective(expr=0)

    blks[0].fs.windpower.system_capacity.setub(wind_ub_mw * 1e3)
    # blks[0].fs.battery.initial_state_of_charge.fix(0)
    blks[0].fs.battery.initial_energy_throughput.fix(0)

    opt = pyo.SolverFactory('ipopt')
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
    h2_turbine_elec = []
    h2_revenue = []
    elec_revenue = []

    for week in range(n_weeks):
        # print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        # opt.options['bound_push'] = 10e-9
        opt.options['max_iter'] = 100000
        # opt.options['tol'] = 1e-6
        # opt.options["halt_on_ampl_error"] = "yes"

        if verbose:
            solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                               tag="properties")
            log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
            log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
            # log_close_to_bounds(m, logger=solve_log)

            # print("Badly scaled variables before solve:")
            # for v, sv in iscale.badly_scaled_var_generator(m, large=1e2, small=1e-2, zero=1e-12):
            #     print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")
        ok = False
        try:
            res = opt.solve(m, tee=verbose, symbolic_solver_labels=True)
            ok = res.Solver.status == 'ok'
        except:
            pass

        # if ok:
        #     m.del_component('obj')
        #     m.obj = pyo.Objective(expr=-m.NPV)
        #     res = opt.solve(m, tee=verbose, symbolic_solver_labels=True)
        #     ok = res.Solver.status == 'ok'

        if verbose and not ok:
            solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO,
                                                tag="properties")
            log_infeasible_constraints(m, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
            log_infeasible_bounds(m, logger=solve_log, tol=1e-4)
            # log_close_to_bounds(m, logger=solve_log)
            blks[0].display()

            # print("Badly scaled variables after solve:")
            # for v, sv in iscale.badly_scaled_var_generator(m, large=1e2, small=1e-2, zero=1e-12):
            #     print(f"    {v} -- {sv} -- {iscale.get_scaling_factor(v)}")


        h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600) for i in range(n_time_points)])
        h2_tank_in.append([pyo.value(blks[i].fs.h2_tank.inlet.flow_mol[0] * 3600) for i in range(n_time_points)])
        h2_tank_out.append([pyo.value(blks[i].fs.h2_tank.outlet.flow_mol[0] * 3600) for i in range(n_time_points)])
        h2_tank_holdup.append(
            [pyo.value(blks[i].fs.h2_tank.material_holdup[0, ('Vap', 'hydrogen')]) for i in range(n_time_points)])
        wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])
        wind_to_grid.append([pyo.value(blks[i].fs.wind_to_grid[0]) for i in range(n_time_points)])
        wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)])
        batt_to_grid.append([pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)])
        wind_to_batt.append([pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)])
        h2_turbine_elec.append([pyo.value(blks[i].fs.h2_turbine.electricity[0]) for i in range(n_time_points)])
        soc.append([pyo.value(blks[i].fs.battery.state_of_charge[0] * 1e-3) for i in range(n_time_points)])
        elec_revenue.append([pyo.value(blks[i].profit) for i in range(n_time_points)])
        h2_revenue.append([pyo.value(m.h2_price_per_kg * blks[i].fs.pem.outlet_state[0].flow_mol / h2_mols_per_kg
                                     * 3600) for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points * n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()
    h2_tank_in = np.asarray(h2_tank_in[0:n_weeks_to_plot]).flatten()
    h2_tank_out = np.asarray(h2_tank_out[0:n_weeks_to_plot]).flatten()
    h2_tank_holdup = np.asarray(h2_tank_holdup[0:n_weeks_to_plot]).flatten()
    batt_out = np.asarray(batt_to_grid[0:n_weeks_to_plot]).flatten()
    batt_in = np.asarray(wind_to_batt[0:n_weeks_to_plot]).flatten()
    batt_soc = np.asarray(soc[0:n_weeks_to_plot]).flatten()
    h2_turbine_elec = np.asarray(soc[0:n_weeks_to_plot]).flatten()
    h2_revenue = np.asarray(h2_revenue[0:n_weeks_to_plot]).flatten()
    elec_revenue = np.asarray(elec_revenue[0:n_weeks_to_plot]).flatten()

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3
    batt_cap = value(blks[0].fs.battery.nameplate_power) * 1e-3
    pem_cap = value(m.pem_system_capacity) * 1e-3
    tank_vol = value(m.h2_tank_volume)
    turb_cap = value(m.turb_system_capacity) * 1e-3

    print("wind mw", wind_cap)
    print("batt mw", batt_cap)
    print("pem mw", pem_cap)
    print("tank m^3", tank_vol)
    print("turb mw", turb_cap)
    if h2_contract:
        print("h2 contract", value(m.contract_capacity))
    print("h2 rev week", value(m.hydrogen_revenue))
    print("elec rev week", value(sum([blk.profit for blk in blks])))
    print("annual rev", value(m.annual_revenue))
    print("npv", value(m.NPV))

    fig, ax1 = plt.subplots(3, 1, figsize=(12, 8))
    plt.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(batt_cap, 2)} MW Battery, "
                 f"{round(pem_cap, 2)} MW PEM, {round(tank_vol, 2)} m^3 Tank and {round(turb_cap, 2)} MW Turbine")

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
    ax1[0].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    ax1[0].minorticks_on()
    ax1[0].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)

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

    ax1[1].tick_params(axis='y', )
    ax1[1].legend()
    ax1[1].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    ax1[1].minorticks_on()
    ax1[1].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)

    ax2 = ax1[1].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array[0:len(hours)], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1[2].set_xlabel('Hour')
    ax1[2].step(hours, elec_revenue, label="Elec rev")
    ax1[2].step(hours, h2_revenue, label="H2 rev")
    ax1[2].step(hours, np.cumsum(elec_revenue), label="Elec rev cumulative")
    ax1[2].step(hours, np.cumsum(h2_revenue), label="H2 rev cumulative")
    ax1[2].legend()
    ax1[2].grid(b=True, which='major', color='k', linestyle='--', alpha=0.2)
    ax1[2].minorticks_on()
    ax1[2].grid(b=True, which='minor', color='k', linestyle='--', alpha=0.2)

    plt.show()

    return wind_cap, batt_cap, pem_cap, tank_vol, turb_cap, value(m.hydrogen_revenue), value(sum([blk.profit for blk in blks])), value(m.NPV)


if __name__ == "__main__":
    wind_battery_pem_tank_turb_optimize(True)
