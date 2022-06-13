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
import pyomo.environ as pyo
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.load_parameters import *

design_opt = True
extant_wind = True


def wind_pem_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    if not extant_wind and design_opt:
        return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
    return []


def wind_pem_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    if not extant_wind and design_opt:
        return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
    return []


def wind_pem_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=wind_op_cost,
        doc="fixed cost of operating wind plant $/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr"
    )
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
    m.fs.splitter.pem_elec[0].fix(0)
    m.fs.splitter.initialize()
    m.fs.splitter.pem_elec[0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_pem)

    m.fs.pem.initialize(outlvl=outlvl)
    if verbose:
        m.fs.pem.report(dof=True)


def wind_pem_model(wind_resource_config, verbose):
    m = create_model(fixed_wind_mw, pem_bar, None, None, None, None, wind_resource_config=wind_resource_config, verbose=verbose)
    if design_opt and not extant_wind:
        m.fs.windpower.system_capacity.unfix()

    initialize_mp(m, verbose=verbose)
    wind_pem_om_costs(m)

    return m


def wind_pem_mp_block(wind_resource_config):
    m = wind_pem_model(wind_resource_config)
    return m


def wind_pem_optimize(n_time_points, h2_price=h2_price_per_kg, verbose=False):
    # create the multiperiod model object
    mp_wind_pem = MultiPeriodModel(n_time_points=n_time_points,
                                   process_model_func=partial(wind_pem_model, verbose=verbose),
                                   linking_variable_func=wind_pem_variable_pairs,
                                   periodic_variable_func=wind_pem_periodic_variable_pairs)

    mp_wind_pem.build_multi_period_model(wind_resource)

    m = mp_wind_pem.pyomo_model
    blks = mp_wind_pem.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=h2_price, mutable=True)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=fixed_pem_mw * 1e3, units=pyunits.kW)
    if not design_opt:
        m.pem_system_capacity.fix(fixed_pem_mw * 1e3)

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_pem = blk.fs.pem
        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
        )
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*blk.fs.splitter.grid_elec[0] * 1e-3
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost - blk_pem.op_total_cost)
        blk.hydrogen_revenue = Expression(expr=m.h2_price_per_kg * blk_pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600)

    # sizing constraint
    m.pem_max_p = Constraint(mp_wind_pem.pyomo_model.TIME,
                                rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)

    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(prices_used[i])

    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if extant_wind:
        m.wind_cap_cost.set_value(0.)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)

    n_weeks = n_time_points / (7 * 24)

    m.annual_revenue = Expression(expr=(sum([blk.profit + blk.hydrogen_revenue for blk in blks])) * 52 / n_weeks)

    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity +
                            m.pem_cap_cost * m.pem_system_capacity) +
                          PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)

    opt = pyo.SolverFactory('ipopt')
    opt.options['max_iter'] = 10000
    h2_prod = []
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []
    h2_revenue = []
    elec_revenue = []

    opt.solve(m, tee=verbose)
    h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600 / 500) for i in range(n_time_points)])
    wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])
    wind_to_grid.append([pyo.value(blks[i].fs.splitter.grid_elec[0]) for i in range(n_time_points)])
    wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)])
    elec_revenue.append([pyo.value(blks[i].profit) for i in range(n_time_points)])
    h2_revenue.append([pyo.value(blks[i].hydrogen_revenue) for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()
    h2_revenue = np.asarray(h2_revenue[0:n_weeks_to_plot]).flatten()
    elec_revenue = np.asarray(elec_revenue[0:n_weeks_to_plot]).flatten()

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3
    pem_cap = value(m.pem_system_capacity) * 1e-3

    fig, ax1 = plt.subplots(3, 1, figsize=(12, 8))
    plt.suptitle(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(pem_cap, 2)} MW PEM")

    # color = 'tab:green'
    ax1[0].set_xlabel('Hour')
    ax1[0].set_ylabel('kW', )
    ax1[0].step(hours, wind_gen, label="Wind Generation")
    ax1[0].step(hours, wind_out, label="Wind to Grid")
    ax1[0].step(hours, wind_to_pem, label="Wind to Pem")
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
    
    # plt.show()

    print("wind mw", wind_cap)
    print("pem mw", pem_cap)
    print("h2 rev", sum(h2_revenue))
    print("elec rev", sum(elec_revenue))
    print("npv", value(m.NPV))

    return wind_cap, pem_cap, sum(h2_revenue), sum(elec_revenue), value(m.NPV)


if __name__ == "__main__":
    wind_pem_optimize(n_time_points=7 * 24)

