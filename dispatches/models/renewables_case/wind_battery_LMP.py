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

pyo_model = None


def wind_battery_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    pairs = [
        (m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
        (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput),
    ]
    if design_opt:
        pairs += [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [
                (m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)
            ]
    return pairs


def wind_battery_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    pairs = [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge)]
    if design_opt:
        pairs += [(m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
        if not extant_wind:
            pairs += [
                (m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)
            ]
    return pairs


def wind_battery_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=wind_op_cost, doc="fixed cost of operating wind plant $10/kW-yr"
    )
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr",
    )


def initialize_mp(m, verbose=False):
    outlvl = idaeslog.INFO if verbose else idaeslog.WARNING

    m.fs.windpower.initialize(outlvl=outlvl)

    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.battery_elec[0].fix(1)
    m.fs.splitter.initialize()
    m.fs.splitter.battery_elec[0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_battery)
    m.fs.battery.elec_in[0].fix()
    m.fs.battery.elec_out[0].fix(value(m.fs.battery.elec_in[0]))
    m.fs.battery.initialize(outlvl=outlvl)
    m.fs.battery.elec_in[0].unfix()
    m.fs.battery.elec_out[0].unfix()
    if verbose:
        m.fs.battery.report(dof=True)


def wind_battery_model(wind_resource_config, verbose=False):
    m = create_model(
        fixed_wind_mw,
        None,
        fixed_batt_mw,
        None,
        None,
        None,
        wind_resource_config=wind_resource_config,
        verbose=verbose,
    )

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    initialize_mp(m, verbose=verbose)

    wind_battery_om_costs(m)
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


def wind_battery_mp_block(wind_resource_config, verbose=False):
    global pyo_model
    if pyo_model is None:
        pyo_model = wind_battery_model(wind_resource_config, verbose=verbose)
    m = pyo_model.clone()
    m.fs.windpower.config.resource_probability_density = wind_resource_config['resource_probability_density']
    m.fs.windpower.setup_resource()
    return m


def wind_battery_optimize(n_time_points, verbose=False):
    # create the multiperiod model object
    mp_wind_battery = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_mp_block, verbose=verbose),
        linking_variable_func=wind_battery_variable_pairs,
        periodic_variable_func=wind_battery_periodic_variable_pairs,
    )

    mp_wind_battery.build_multi_period_model(wind_resource)

    m = mp_wind_battery.pyomo_model
    blks = mp_wind_battery.get_active_process_blocks()

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = (
            blk.lmp_signal * (blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0])
        )
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost)

    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(prices_used[i] * 1e-3)
    
    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if extant_wind:
        m.wind_cap_cost.set_value(0.0)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)

    n_weeks = n_time_points / (7 * 24)
    m.annual_revenue = Expression(expr=sum([blk.profit for blk in blks]) * 52 / n_weeks)
    m.NPV = Expression(
        expr=-(
            m.wind_cap_cost * blks[0].fs.windpower.system_capacity
            + m.batt_cap_cost * blks[0].fs.battery.nameplate_power
        )
        + PA * m.annual_revenue
    )
    m.obj = pyo.Objective(expr=-m.NPV * 1e-5)

    blks[0].fs.windpower.system_capacity.setub(wind_ub_mw * 1e3)
    blks[0].fs.battery.initial_state_of_charge.fix(0)
    blks[0].fs.battery.initial_energy_throughput.fix(0)

    opt = pyo.SolverFactory("ipopt")
    opt.solve(m, tee=verbose)

    return mp_wind_battery


def record_results(mp_wind_battery):

    m = mp_wind_battery.pyomo_model

    batt_to_grid = []
    wind_to_grid = []
    wind_to_batt = []
    wind_gen = []
    soc = []
    elec_revenue = []

    blks = mp_wind_battery.get_active_process_blocks()
    n_time_points = len(blks)
    soc = [
        pyo.value(blks[i].fs.battery.state_of_charge[0]) for i in range(n_time_points)
    ]
    wind_gen = [
        pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)
    ]
    batt_to_grid = [
        pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)
    ]
    wind_to_grid = [pyo.value(blks[i].fs.splitter.grid_elec[0]) for i in range(n_time_points)]
    wind_to_batt = [
        pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)
    ]
    elec_revenue = [pyo.value(blks[i].profit) for i in range(n_time_points)]
    lmp = [pyo.value(blks[i].lmp_signal) for i in range(n_time_points)]

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3
    batt_cap = value(blks[0].fs.battery.nameplate_power) * 1e-3

    annual_revenue = value(m.annual_revenue)
    npv = value(m.NPV)

    print("wind mw", wind_cap)
    print("batt mw", batt_cap)
    print("elec rev", sum(elec_revenue))
    print("annual rev", annual_revenue)
    print("npv", npv)

    return (
        soc,
        wind_gen,
        batt_to_grid,
        wind_to_grid,
        wind_to_batt,
        elec_revenue,
        lmp,
        wind_cap,
        batt_cap,
        annual_revenue,
        npv,
    )


def plot_results(
    soc,
    wind_gen,
    batt_to_grid,
    wind_to_grid,
    wind_to_batt,
    elec_revenue,
    lmp,
    wind_cap,
    batt_cap,
    annual_revenue,
    npv,
):

    hours = [t for t in range(len(soc))]

    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))
    plt.suptitle(
        f"Optimal NPV ${round(npv * 1e-6)}mil from {round(batt_cap, 2)} MW Battery"
    )

    # color = 'tab:green'
    ax1[0].set_xlabel("Hour")
    ax1[0].set_ylabel(
        "kW",
    )
    ax1[0].step(hours, wind_gen, label="Wind Generation [kW]")
    ax1[0].step(hours, wind_to_grid, label="Wind to Grid [kW]")
    ax1[0].step(hours, wind_to_batt, label="Wind to Batt [kW]")
    ax1[0].step(hours, batt_to_grid, label="Batt to Grid [kW]")
    ax1[0].tick_params(
        axis="y",
    )
    ax1[0].legend()
    ax1[0].grid(visible=True, which="major", color="k", linestyle="--", alpha=0.2)
    ax1[0].minorticks_on()
    ax1[0].grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.2)

    ax2 = ax1[0].twinx()
    color = "k"
    ax2.set_ylabel("LMP [$/MWh]", color=color)
    ax2.plot(hours, lmp, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax2 = ax1[1].twinx()
    color = "k"
    ax2.set_ylabel("LMP [$/MWh]", color=color)
    ax2.plot(hours, lmp, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax1[1].set_xlabel("Hour")
    ax1[1].step(hours, elec_revenue, label="Elec rev")
    ax1[1].step(hours, np.cumsum(elec_revenue), label="Elec rev cumulative")
    ax1[1].legend()
    ax1[1].grid(visible=True, which="major", color="k", linestyle="--", alpha=0.2)
    ax1[1].minorticks_on()
    ax1[1].grid(visible=True, which="minor", color="k", linestyle="--", alpha=0.2)
    # plt.show()

    return ax1, ax2


if __name__ == "__main__":
    mp_wind_battery = wind_battery_optimize(n_time_points=7 * 24)
    soc, wind_gen, batt_to_grid, wind_to_grid, wind_to_batt, elec_revenue, lmp, wind_cap, batt_cap, annual_revenue, npv = record_results(mp_wind_battery)
    ax1, ax2 = plot_results(soc, wind_gen, batt_to_grid, wind_to_grid, wind_to_batt, elec_revenue, lmp, wind_cap, batt_cap, annual_revenue, npv)
    plt.show()
