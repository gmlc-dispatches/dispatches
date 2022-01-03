import pyomo.environ as pyo
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_parameters import *

design_opt = True
extant_wind = True


def wind_battery_variable_pairs(m1, m2):
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
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
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
            pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]
    return pairs


def wind_battery_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=wind_op_cost,
        doc="fixed cost of operating wind plant $10/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr"
    )


def initialize_mp(m, verbose=False):
    m.fs.windpower.initialize()

    propagate_state(m.fs.wind_to_splitter)
    m.fs.splitter.split_fraction['grid', 0].fix(1)
    m.fs.splitter.initialize()
    m.fs.splitter.split_fraction['grid', 0].unfix()
    if verbose:
        m.fs.splitter.report(dof=True)

    propagate_state(m.fs.splitter_to_grid)

    propagate_state(m.fs.splitter_to_battery)
    m.fs.battery.elec_in[0].fix()
    m.fs.battery.elec_out[0].fix(value(m.fs.battery.elec_in[0]))
    m.fs.battery.initialize()
    m.fs.battery.elec_in[0].unfix()
    m.fs.battery.elec_out[0].unfix()
    if verbose:
        m.fs.battery.report(dof=True)


def wind_battery_model(wind_resource_config):
    m = create_model(fixed_wind_mw, None, fixed_batt_mw, None, None, None, wind_resource_config=wind_resource_config)

    m.fs.battery.initial_state_of_charge.fix(0)
    m.fs.battery.initial_energy_throughput.fix(0)
    initialize_mp(m, verbose=False)

    wind_battery_om_costs(m)
    m.fs.battery.initial_state_of_charge.unfix()
    m.fs.battery.initial_energy_throughput.unfix()

    if design_opt:
        if not extant_wind:
            m.fs.windpower.system_capacity.unfix()
        m.fs.battery.nameplate_power.unfix()
    return m


def wind_battery_mp_block(wind_resource_config):
    m = wind_battery_model(wind_resource_config)
    batt = m.fs.battery

    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge <= battery_ramp_rate)
    return m


def wind_battery_optimize():
    # create the multiperiod model object
    mp_wind_battery = MultiPeriodModel(n_time_points=n_time_points,
                                       process_model_func=wind_battery_mp_block,
                                       linking_variable_func=wind_battery_variable_pairs,
                                       periodic_variable_func=wind_battery_periodic_variable_pairs)

    mp_wind_battery.build_multi_period_model(wind_resource)

    m = mp_wind_battery.pyomo_model
    blks = mp_wind_battery.get_active_process_blocks()

    #add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*(blk.fs.wind_to_grid[0] + blk_battery.elec_out[0])
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost)

    m.wind_cap_cost = pyo.Param(default=wind_cap_cost, mutable=True)
    if extant_wind:
        m.wind_cap_cost.set_value(0.)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)

    n_weeks = 1
    m.annual_revenue = Expression(expr=sum([blk.profit for blk in blks]) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity +
                              m.batt_cap_cost * blks[0].fs.battery.nameplate_power) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)

    blks[0].fs.windpower.system_capacity.setub(wind_ub_mw * 1e3)
    blks[0].fs.battery.initial_state_of_charge.fix(0)
    blks[0].fs.battery.initial_energy_throughput.fix(0)

    opt = pyo.SolverFactory('ipopt')
    opt.options['bound_push'] = 10e-10

    opt.options['max_iter'] = 10000
    batt_to_grid = []
    wind_to_grid = []
    wind_to_batt = []
    wind_gen = []
    soc = []

    for week in range(n_weeks):
        print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        opt.solve(m, tee=True)
        soc.append([pyo.value(blks[i].fs.battery.state_of_charge[0]) * 1e-3 for i in range(n_time_points)])
        wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) * 1e-3 for i in range(n_time_points)])
        batt_to_grid.append([pyo.value(blks[i].fs.battery.elec_out[0]) * 1e-3 for i in range(n_time_points)])
        wind_to_grid.append([pyo.value(blks[i].fs.wind_to_grid[0]) * 1e-3 for i in range(n_time_points)])
        wind_to_batt.append([pyo.value(blks[i].fs.battery.elec_in[0]) * 1e-3 for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    batt_out = np.asarray(batt_to_grid[0:n_weeks_to_plot]).flatten()
    batt_in = np.asarray(wind_to_batt[0:n_weeks_to_plot]).flatten()
    batt_soc = np.asarray(soc[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()

    wind_cap = value(blks[0].fs.windpower.system_capacity) * 1e-3
    batt_cap = value(blks[0].fs.battery.nameplate_power) * 1e-3

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # print(batt_in)
    # print(batt_out)
    # print(wind_out)

    # color = 'tab:green'
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('MW', )
    ax1.step(hours, wind_gen)
    ax1.set_ylabel('MW', )
    ax1.step(hours, wind_gen, label="Wind Generation")
    ax1.step(hours, wind_out, label="Wind to Grid")
    ax1.step(hours, batt_in, label="Wind to Batt")
    ax1.step(hours, batt_out, label="Batt to Grid")
    ax1.step(hours, batt_soc, label="Batt SOC")
    ax1.tick_params(axis='y', )
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'grey'
    ax2.set_ylabel('$/MWh', color=color)
    ax2.plot(hours, lmp_array, color=color, linestyle='dotted', label="LMP")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')
    plt.title(f"Optimal NPV ${round(value(m.NPV) * 1e-6)}mil from {round(wind_cap)} MW Wind and {round(batt_cap, 2)} MW Battery")
    plt.show()

    print("wind mw", wind_cap)
    print("batt mw", batt_cap)
    print("annual rev", value(m.annual_revenue))
    print("npv", value(m.NPV))


wind_battery_optimize()
