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
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.load_parameters import *

design_opt = True
extant_wind = True

pyo_model = None


def wind_battery_pem_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    pairs = [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
             (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput)]
    if design_opt:
        pairs += [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity),]
    return pairs


def wind_battery_pem_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    pairs = [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge)]
    if design_opt:
        pairs += [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity), ]
    return pairs


def wind_battery_pem_om_costs(m):
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


def wind_battery_pem_model(wind_resource_config, verbose):
    m = create_model(fixed_wind_mw, pem_bar, fixed_batt_mw, None, None, None,  wind_resource_config=wind_resource_config,
                     verbose=verbose)

    m.fs.pem.outlet_state[0].sum_mole_frac_out.deactivate()
    m.fs.pem.outlet_state[0].component_flow_balances.deactivate()
    m.fs.pem.outlet_state[0].phase_fraction_constraint.deactivate()

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)

    initialize_mp(m, verbose=verbose)

    wind_battery_pem_om_costs(m)
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()

    batt = m.fs.battery

    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge <= battery_ramp_rate)

    if design_opt:
        if not extant_wind:
            m.fs.windpower.system_capacity.unfix()
        m.fs.battery.nameplate_power.unfix()
    return m


def wind_battery_pem_mp_block(wind_resource_config, verbose):
    global pyo_model
    if pyo_model is None:
        pyo_model = wind_battery_pem_model(wind_resource_config, verbose=verbose)
    m = pyo_model.clone()
    m.fs.windpower.config.resource_probability_density = wind_resource_config['resource_probability_density']
    m.fs.windpower.setup_resource()

    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING
    m.fs.windpower.initialize(outlvl=outlvl)
    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.initialize()
    return m


def wind_battery_pem_optimize(time_points, h2_price=h2_price_per_kg, verbose=False, plot=False):
    from timeit import default_timer
    start = default_timer()

    # create the multiperiod model object
    mp_battery_wind_pem = MultiPeriodModel(n_time_points=time_points,
                                           process_model_func=partial(wind_battery_pem_mp_block, verbose=verbose),
                                           linking_variable_func=wind_battery_pem_variable_pairs,
                                           periodic_variable_func=wind_battery_pem_periodic_variable_pairs)

    mp_battery_wind_pem.build_multi_period_model(wind_resource)

    m = mp_battery_wind_pem.pyomo_model
    blks = mp_battery_wind_pem.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=h2_price, mutable=True)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=fixed_pem_mw * 1e3, units=pyunits.kW)
    if not design_opt:
        m.pem_system_capacity.fix(fixed_pem_mw * 1e3)

    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if extant_wind:
        m.wind_cap_cost.set_value(0.)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem

        # add operating costs
        blk_wind.op_total_cost = Expression(
            expr=blk_wind.system_capacity * blk_wind.op_cost / 8760
        )

        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
        )

        # add market data for each block
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal * (blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0]) * 1e-3    # to $/kWh
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost - blk_pem.op_total_cost)
        blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg * blk_pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600)

    # sizing constraints
    m.pem_max_p = Constraint(mp_battery_wind_pem.pyomo_model.TIME,
                             rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)

    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(prices_used[i]) 

    n_weeks = time_points / (7 * 24)

    m.annual_revenue = Expression(expr=(sum([blk.profit + blk.hydrogen_revenue for blk in blks])) * 52 / n_weeks)

    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity +
                              m.batt_cap_cost * blks[0].fs.battery.nameplate_power +
                              m.pem_cap_cost * m.pem_system_capacity) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV * 1e-5)

    blks[0].fs.windpower.system_capacity.setub(wind_ub_mw * 1e3)
    # blks[0].fs.battery.initial_state_of_charge.fix(0)
    blks[0].fs.battery.initial_energy_throughput.fix(0)

    opt = pyo.SolverFactory('ipopt')
    # opt.options['max_iter'] = 50000
    # opt.options['tol'] = 1e-6

    time_to_create_model = default_timer() - start

    # status_obj, solved, iters, time, regu = ipopt_solve_with_stats(m, opt, opt.options['max_iter'], 60*210)
    # solver_res = (status_obj, solved, iters, time, regu)
    ipopt_res = opt.solve(m)

    h2_prod = []
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []
    wind_to_batt = []
    batt_to_grid = []
    soc = []
    h2_revenue = []
    elec_revenue = []

    h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600) for i in range(time_points)])
    wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(time_points)])
    wind_to_grid.append([pyo.value(blks[i].fs.splitter.grid_elec[0]) for i in range(time_points)])
    wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(time_points)])
    batt_to_grid.append([pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(time_points)])
    wind_to_batt.append([pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(time_points)])
    soc.append([pyo.value(blks[i].fs.battery.state_of_charge[0] * 1e-3) for i in range(time_points)])
    elec_revenue.append([pyo.value(blks[i].profit) for i in range(time_points)])
    h2_revenue.append([pyo.value(blks[i].hydrogen_revenue) for i in range(time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(time_points)
    lmp_array = weekly_prices[0:time_points].flatten()
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()
    batt_out = np.asarray(batt_to_grid[0:n_weeks_to_plot]).flatten()
    batt_in = np.asarray(wind_to_batt[0:n_weeks_to_plot]).flatten()
    batt_soc = np.asarray(soc[0:n_weeks_to_plot]).flatten()
    h2_revenue = np.asarray(h2_revenue[0:n_weeks_to_plot]).flatten()
    elec_revenue = np.asarray(elec_revenue[0:n_weeks_to_plot]).flatten()

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3
    batt_cap = value(blks[0].fs.battery.nameplate_power) * 1e-3
    pem_cap = value(m.pem_system_capacity) * 1e-3

    if plot:
        fig, ax1 = plt.subplots(3, 1, figsize=(12, 8))
        plt.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(batt_cap, 2)} MW Battery and "
                     f"{round(pem_cap, 2)} MW PEM")

        # color = 'tab:green'
        ax1[0].set_xlabel('Hour')
        # ax1[0].set_ylabel('MW', )
        ax1[0].step(hours, wind_gen, label="Wind Generation [kW]")
        ax1[0].step(hours, wind_out, label="Wind to Grid [kW]")
        ax1[0].step(hours, batt_in, label="Wind to Batt [kW]")
        ax1[0].step(hours, batt_out, label="Batt to Grid [kW]")
        ax1[0].step(hours, batt_soc, label="Batt SOC [MWh]")
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

        ax1[1].set_xlabel('Hour')
        ax1[1].set_ylabel('kg', )
        # ax1[1].step(hours, wind_gen, label="Wind Generation")
        # ax1[1].step(hours, wind_out, label="Wind to Grid")
        ax1[1].step(hours, h2_prod, label="H2 production")
        # ax1[1].step(hours, wind_to_pem, label="Wind to Pem")
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
        ax1[2].step(hours, elec_revenue, label="Elec rev")
        ax1[2].step(hours, h2_revenue, label="H2 rev")
        ax1[2].step(hours, np.cumsum(elec_revenue), label="Elec rev cumulative")
        ax1[2].step(hours, np.cumsum(h2_revenue), label="H2 rev cumulative")
        ax1[2].legend()
        ax1[2].grid(visible=True, which='major', color='k', linestyle='--', alpha=0.2)
        ax1[2].minorticks_on()
        ax1[2].grid(visible=True, which='minor', color='k', linestyle='--', alpha=0.2)
        plt.show()

    design_res = {
        'wind_mw': wind_cap,
        "batt_mw": batt_cap,
        "pem_mw": pem_cap,
        "annual_rev_h2": sum(h2_revenue) * 52 / n_weeks,
        "annual_rev_E": sum(elec_revenue) * 52 / n_weeks,
        "NPV": value(m.NPV)
    }
    print(design_res)

    return design_res, ipopt_res, time_to_create_model


if __name__ == "__main__":
    wind_battery_pem_optimize(7*24*2, h2_price=h2_price_per_kg, verbose=False, plot=False)
