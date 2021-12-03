import pyomo.environ as pyo
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_LMP import *

design_opt = True


def wind_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]


def wind_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]


def wind_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=43,
        doc="fixed cost of operating wind plant $/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr"
    )


def wind_model(wind_resource_config):
    wind_mw = 200

    m = create_model(wind_mw, None, None, None, None, None, wind_resource_config=wind_resource_config)
    if design_opt:
        m.fs.windpower.system_capacity.unfix()

    # set_initial_conditions(m, pem_bar * 0.1)
    initialize_model(m, verbose=False)
    wind_om_costs(m)

    return m


    # solver = SolverFactory('ipopt')
    # res = solver.solve(m, tee=True)
    # m.fs.h2_turbine.min_p = pyo.Constraint(expr=m.fs.h2_turbine.turbine.work_mechanical[0] >= turb_p_lower_bound * 1e6)
    # m.fs.h2_turbine.max_p = pyo.Constraint(expr=m.fs.h2_turbine.turbine.work_mechanical[0] <= turb_p_upper_bound * 1e6)


def wind_optimize():
    # create the multiperiod model object
    wind = MultiPeriodModel(n_time_points=n_time_points,
                            process_model_func=wind_model,
                            linking_variable_func=wind_variable_pairs,
                            periodic_variable_func=wind_periodic_variable_pairs)

    wind.build_multi_period_model(wind_resource)

    m = wind.pyomo_model
    blks = wind.get_active_process_blocks()

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*blk.fs.windpower.electricity[0]
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost)

    m.wind_cap_cost = pyo.Param(default=1555, mutable=True)

    n_weeks = 1

    m.annual_revenue = Expression(expr=(sum([blk.profit for blk in blks])) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity) +
                          PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)

    # IPOPT can't get to zero thanks to the boundary pushing maybe?
    opt = pyo.SolverFactory('cbc')
    # opt.options['max_iter'] = 10000
    wind_gen = []

    for week in range(n_weeks):
        print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        opt.solve(m, tee=True)
        wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])


    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()


    fig, ax1 = plt.subplots(figsize=(12, 8))

    # color = 'tab:green'
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('kW', )
    ax1.step(hours, wind_gen, label="Wind Generation")
    ax1.tick_params(axis='y', )
    ax1.legend()

    ax2 = ax1.twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend()
    plt.show()

    print(value(blks[0].fs.windpower.system_capacity))
    print(value(m.annual_revenue))
    print(value(m.NPV))


wind_optimize()

# with LMPs as is, no wind is the best option
