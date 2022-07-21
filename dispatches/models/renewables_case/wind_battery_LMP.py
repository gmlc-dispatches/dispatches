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
import pandas as pd
import pyomo.environ as pyo
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *
from dispatches.models.renewables_case.load_parameters import *
from dispatches.models.renewables_case.load_parameters import df


def wind_battery_variable_pairs(m1, m2):
    """
    This function links together unit model state variables from one timestep to the next.

    The battery's state of charge and energy throughput values need to be consistent across time blocks.

    Args:
        m1: current time block model
        m2: next time block model
    """
    pairs = [
        (m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
        (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput),
        (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)
    ]
    return pairs


def wind_battery_periodic_variable_pairs(m1, m2):
    """
    The final battery storage of charge must be the same as in the intial timestep. 

    Args:
        m1: final time block model
        m2: first time block model
    """
    pairs = [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
             (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]
    return pairs


def wind_battery_om_costs(m):
    """
    Add unit fixed and variable operating costs as parameters for the unit model m
    """
    m.fs.windpower.op_cost = pyo.Param(
        initialize=wind_op_cost, doc="fixed cost of operating wind plant $10/kW-yr"
    )
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr",
    )
    m.fs.battery.var_cost = pyo.Expression(
        expr=m.fs.battery.degradation_rate * (m.fs.battery.energy_throughput[0] - m.fs.battery.initial_energy_throughput) * batt_rep_cost_kwh,
        doc="variable operating of the battery $/kWh"
    )


def initialize_mp(m, verbose=False):
    """
    Initializing the flowsheet is done starting with the wind model and propagating the solved initial state to the splitter and battery.

    The splitter is initialized with no flow to the battery so all electricity flows to the grid, which makes the initialization of all
    unit models downstream of the wind plant independent of its time-varying electricity production. This initialzation function can
    then be repeated for all timesteps within a dynamic analysis.

    Args:
        m: model
        verbose:
    """
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


def wind_battery_model(wind_resource_config, input_params, verbose=False):
    """
    Creates an initialized flowsheet model for a single time step with operating, size and cost components
    
    First, the model is created using the input_params and wind_resource_config
    Second, the model is initialized so that it solves and its values are internally consistent
    Third, battery ramp constraints and operating cost components are added

    Args:
        wind_resource_config: wind resource for the time step
        input_params: size and operation parameters. Required keys: `wind_mw` and `batt_mw`
        verbose:
    """
    m = create_model(
        input_params['wind_mw'],
        None,
        input_params['batt_mw'],
        None,
        None,
        None,
        wind_resource_config=wind_resource_config
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

    return m


def wind_battery_mp_block(wind_resource_config, input_params, verbose=False):
    """
    Wrapper of `wind_battery_model` for creating the process model per time point for the MultiPeriod model.
    Uses cloning of the Pyomo model in order to reduce runtime. 
    The clone is reinitialized with the `wind_resource_config` for the given time point, which only required modifying
    the windpower and the splitter, as the rest of the units have no flow and therefore is unaffected by wind resource changes.

    Args:
        wind_resource_config: dictionary with `resource_speed` for the time step
        input_params: size and operation parameters. Required keys: `wind_mw`, `batt_mw`
        verbose:
    """
    if 'pyo_model' not in input_params.keys():
        input_params['pyo_model'] = wind_battery_model(wind_resource_config, input_params, verbose=verbose)
    m = input_params['pyo_model'].clone()
    if 'resource_speed' in wind_resource_config.keys():
        m.fs.windpower.config.resource_speed = wind_resource_config['resource_speed']
    elif 'capacity_factor' in wind_resource_config.keys():
        m.fs.windpower.config.capacity_factor = wind_resource_config['capacity_factor']
    else:
        raise ValueError(f"`wind_resource_config` dict must contain either 'resource_speed' or 'capacity_factor' values")
    m.fs.windpower.setup_resource()
    return m


def wind_battery_optimize(n_time_points, input_params, verbose=False):
    """
    The main function for optimizing the flowsheet's design and operating variables for Net Present Value. 

    Creates the MultiPeriodModel and adds the size and operating constraints in addition to the Net Present Value Objective.
    The NPV is a function of the capital costs, the electricity market profit and the capital recovery factor.
    The operating decisions and state evolution of the unit models and the flowsheet as a whole form the constraints of the Linear Program.

    Required input parameters include:
        `wind_mw`: initial guess of the wind size
        `wind_mw_ub`: upper bound of wind size
        `batt_mw`: initial guess of the battery size
        `wind_resource`: dictionary of wind resource configs for each time point
        `DA_LMPs`: LMPs for each time point
        `design_opt`: true to optimize design, else sizes are fixed at initial guess sizes
        `extant_wind`: if true, fix wind size to initial size and do not add wind capital cost to NPV

    Args:
        n_time_points: number of periods in MultiPeriod model
        input_params: 
        verbose: print all logging and outputs from unit models, initialization, solvers, etc
        plot: plot the operating variables time series
    """
    mp_wind_battery = MultiPeriodModel(
        n_time_points=n_time_points,
        process_model_func=partial(wind_battery_mp_block, input_params=input_params, verbose=verbose),
        linking_variable_func=wind_battery_variable_pairs,
        periodic_variable_func=wind_battery_periodic_variable_pairs,
    )

    mp_wind_battery.build_multi_period_model(input_params['wind_resource'])

    m = mp_wind_battery.pyomo_model
    blks = mp_wind_battery.get_active_process_blocks()
    blks[0].fs.battery.initial_state_of_charge.fix(0)
    blks[0].fs.battery.initial_energy_throughput.fix(0)

    m.wind_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['wind_mw'] * 1e3, units=pyunits.kW, bounds=(0, input_params['wind_mw_ub'] * 1e3))
    m.battery_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['batt_mw'] * 1e3, units=pyunits.kW)
    
    if input_params['design_opt']:
        for blk in blks:
            if not input_params['extant_wind']:
                blk.fs.windpower.system_capacity.unfix()
            blk.fs.battery.nameplate_power.unfix()
    
    m.wind_max_p = Constraint(mp_wind_battery.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.windpower.system_capacity <= m.wind_system_capacity)
    m.battery_max_p = Constraint(mp_wind_battery.pyomo_model.TIME, rule=lambda b, t: blks[t].fs.battery.nameplate_power <= m.battery_system_capacity)

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        
        # add operating costs
        blk_wind.op_total_cost = Expression(
            expr=blk_wind.system_capacity * blk_wind.op_cost / 8760
        )

        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = (
            blk.lmp_signal * (blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0])
        )
        blk.profit = pyo.Expression(expr=blk.revenue 
                                         - blk_wind.op_total_cost
                                         - blk_battery.var_cost)

    for (i, blk) in enumerate(blks):
        blk.lmp_signal.set_value(input_params['DA_LMPs'][i] * 1e-3) 
    
    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if input_params['extant_wind']:
        m.wind_cap_cost.set_value(0.0)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)

    n_weeks = n_time_points / (7 * 24)
    m.annual_revenue = Expression(expr=sum([blk.profit for blk in blks]) * 52 / n_weeks)
    m.NPV = Expression(
        expr=-(
            m.wind_cap_cost * m.wind_system_capacity
            + m.batt_cap_cost * m.battery_system_capacity
        )
        + PA * m.annual_revenue
    )
    m.obj = pyo.Objective(expr=-m.NPV * 1e-5)

    opt = pyo.SolverFactory("cbc")
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
        pyo.value(blks[i].fs.windpower.electricity[0]) * 1e-3 for i in range(n_time_points)
    ]
    batt_to_grid = [
        pyo.value(blks[i].fs.battery.elec_out[0]) * 1e-3 for i in range(n_time_points)
    ]
    wind_to_grid = [pyo.value(blks[i].fs.splitter.grid_elec[0]) * 1e-3 for i in range(n_time_points)]
    wind_to_batt = [
        pyo.value(blks[i].fs.battery.elec_in[0]) * 1e-3 for i in range(n_time_points)
    ]
    elec_revenue = [pyo.value(blks[i].profit) for i in range(n_time_points)]
    lmp = [pyo.value(blks[i].lmp_signal) * 1e3 for i in range(n_time_points)]

    wind_cap = value(m.wind_system_capacity) * 1e-3
    batt_cap = value(m.battery_system_capacity) * 1e-3

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
        "MW",
    )
    ax1[0].step(hours, wind_gen, label="Wind Generation [MW]")
    ax1[0].step(hours, wind_to_grid, label="Wind to Grid [MW]")
    ax1[0].step(hours, wind_to_batt, label="Wind to Batt [MW]")
    ax1[0].step(hours, batt_to_grid, label="Batt to Grid [MW]")
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
    mp_wind_battery = wind_battery_optimize(n_time_points=6 * 24, input_params=default_input_params)
    soc, wind_gen, batt_to_grid, wind_to_grid, wind_to_batt, elec_revenue, lmp, wind_cap, batt_cap, annual_revenue, npv = record_results(mp_wind_battery)
    ax1, ax2 = plot_results(soc, wind_gen, batt_to_grid, wind_to_grid, wind_to_batt, elec_revenue, lmp, wind_cap, batt_cap, annual_revenue, npv)
    # plt.show()
