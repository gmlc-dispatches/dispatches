import copy
import pyomo.environ as pyo
import numpy as np
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *


def wind_battery_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    return [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
            (m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity),
            (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)]


def wind_battery_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    return [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
            (m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity),
            (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power)
            ]


def wind_battery_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=43,
        doc="profit of operating wind plant $10/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total cooling water profit in $/hr"
    )


def wind_battery_model():
    wind_mw = 200
    pem_bar = 8
    batt_mw = 100
    valve_cv = 0.0001
    tank_len_m = 0.1
    turb_p_lower_bound = 300
    turb_p_upper_bound = 450

    # m = create_model(wind_mw, pem_bar, batt_mw, valve_cv, tank_len_m)
    m = create_model(wind_mw, None, batt_mw, None, None)
    m.fs.windpower.system_capacity.unfix()
    m.fs.battery.nameplate_power.unfix()

    # set_initial_conditions(m, pem_bar * 0.1)
    initialize_model(m, verbose=False)
    wind_battery_om_costs(m)

    return m


    # solver = SolverFactory('ipopt')
    # res = solver.solve(m, tee=True)
    # m.fs.h2_turbine.min_p = pyo.Constraint(expr=m.fs.h2_turbine.turbine.work_mechanical[0] >= turb_p_lower_bound * 1e6)
    # m.fs.h2_turbine.max_p = pyo.Constraint(expr=m.fs.h2_turbine.turbine.work_mechanical[0] <= turb_p_upper_bound * 1e6)



with open('/Users/dguittet/Projects/Dispatches/idaes-pse/idaes/apps/multiperiod/examples/rts_results_all_prices.npy', 'rb') as f:
    dispatch = np.load(f)
    price = np.load(f)

prices_used = copy.copy(price)
prices_used[prices_used > 200] = 200
weekly_prices = prices_used.reshape(52, 168)

# simple financial assumptions
i = 0.05    # discount rate
N = 30      # years
PA = ((1+i)**N - 1)/(i*(1+i)**N)    # present value / annuity


def wind_battery_mp_block():
    battery_ramp_rate = 50
    m = wind_battery_model()
    batt = m.fs.batt

    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge[0] - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge[0] <= battery_ramp_rate)
    return m


n_time_points = 7*24    # hours in a week
# n_time_points = 7


def wind_battery_optimize():
    # create the multiperiod model object
    mp_wind_battery = MultiPeriodModel(n_time_points=n_time_points,
                                       process_model_func=wind_battery_model,
                                       linking_variable_func=wind_battery_variable_pairs,
                                       periodic_variable_func=wind_battery_periodic_variable_pairs)

    mp_wind_battery.build_multi_period_model()

    m = mp_wind_battery.pyomo_model
    blks = mp_wind_battery.get_active_process_blocks()

    #add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*(blk.fs.wind_to_grid[0] + blk_battery.elec_out[0])
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost)

    m.wind_cap_cost = pyo.Param(default=1555, mutable=True)
    m.batt_cap_cost = pyo.Param(default=1000 + 500 * 4, mutable=True)

    n_weeks = 1
    m.annual_revenue = Expression(expr=sum([blk.profit for blk in blks]) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity +
                            m.batt_cap_cost * blks[0].fs.battery.nameplate_power) +
                          PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)
    blks[0].fs.battery.initial_state_of_charge.fix(0)

    opt = pyo.SolverFactory('ipopt')
    batt_to_grid = []
    wind_to_grid = []
    wind_to_batt = []

    for week in range(n_weeks):
        print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        opt.solve(m, tee=False)
        batt_to_grid.append([pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)])
        wind_to_grid.append([pyo.value(blks[i].fs.wind_to_grid[0]) for i in range(n_time_points)])
        wind_to_batt.append([pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)])


    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    batt_out = np.asarray(batt_to_grid[0:n_weeks_to_plot]).flatten()
    batt_in = np.asarray(wind_to_batt[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()


    fig, ax1 = plt.subplots(figsize=(12, 8))

    print(batt_in)
    print(batt_out)
    print(wind_out)

    # color = 'tab:green'
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('kW', )
    ax1.step(hours, wind_out, label="Wind to Grid")
    ax1.step(hours, batt_in, label="Wind to Batt")
    ax1.step(hours, batt_out, label="Batt to Grid")
    ax1.tick_params(axis='y', )
    ax1.legend()

    ax2 = ax1.twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend()
    plt.show()

    print(value(blks[0].fs.windpower.system_capacity))
    print(value(blks[0].fs.battery.nameplate_power))
    print(value(m.annual_revenue))
    print(value(m.NPV))


wind_battery_optimize()