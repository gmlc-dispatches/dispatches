import pyomo.environ as pyo
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_LMP import *


def wind_pem_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]


def wind_pem_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    return [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity)]


def wind_pem_om_costs(m):
    m.fs.windpower.op_cost = pyo.Param(
        initialize=43,
        doc="fixed cost of operating wind plant $/kW-yr")
    m.fs.windpower.op_total_cost = Expression(
        expr=m.fs.windpower.system_capacity * m.fs.windpower.op_cost / 8760,
        doc="total fixed cost of wind in $/hr"
    )
    m.fs.pem.op_cost = pyo.Param(
        initialize=47.9,
        doc="fixed cost of operating pem $/kW-yr"
    )
    m.fs.pem.var_cost = pyo.Param(
        initialize=1.3/1000,
        doc="variable cost of pem $/kW"
    )
    # m.fs.pem.op_total_cost = Expression(
    #     expr=m.fs.pem.system_capacity * m.fs.pem.op_cost / 8760 + m.fs.pem.var_cost * m.fs.pem.electricity[0],
    #     doc="total fixed cost of pem in $/hr"
    # )


def wind_pem_model(wind_resource_config):
    wind_mw = 200
    pem_bar = 8
    batt_mw = 100
    valve_cv = 0.0001
    tank_len_m = 0.1
    turb_p_lower_bound = 300
    turb_p_upper_bound = 450

    # m = create_model(wind_mw, pem_bar, batt_mw, valve_cv, tank_len_m)
    m = create_model(wind_mw, pem_bar, None, None, None, wind_resource_config=wind_resource_config)
    # m.fs.windpower.system_capacity.unfix()

    # set_initial_conditions(m, pem_bar * 0.1)
    initialize_model(m, verbose=False)
    wind_pem_om_costs(m)

    return m


    # solver = SolverFactory('ipopt')
    # res = solver.solve(m, tee=True)
    # m.fs.h2_turbine.min_p = pyo.Constraint(expr=m.fs.h2_turbine.turbine.work_mechanical[0] >= turb_p_lower_bound * 1e6)
    # m.fs.h2_turbine.max_p = pyo.Constraint(expr=m.fs.h2_turbine.turbine.work_mechanical[0] <= turb_p_upper_bound * 1e6)


def wind_pem_mp_block(wind_resource_config):
    m = wind_pem_model(wind_resource_config)
    return m


def wind_pem_optimize():
    # create the multiperiod model object
    mp_pem_battery = MultiPeriodModel(n_time_points=n_time_points,
                                      process_model_func=wind_pem_model,
                                      linking_variable_func=wind_pem_variable_pairs,
                                      periodic_variable_func=wind_pem_periodic_variable_pairs)

    mp_pem_battery.build_multi_period_model(wind_resource)

    m = mp_pem_battery.pyomo_model
    blks = mp_pem_battery.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=1.9, mutable=True)
    h2_mols_per_kg = 500
    # m.contract_capacity = Var(domain=NonNegativeReals, initialize=100, units=pyunits.mol/pyunits.second)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=20, units=pyunits.kW)
    m.pem_system_capacity.fix(20)


    #add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_pem = blk.fs.pem
        blk_pem.max_p = Constraint(blk_pem.flowsheet().config.time,
                                 rule=lambda b, t: b.electricity[t] <= m.pem_system_capacity)
        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
            doc="total fixed cost of pem in $/hr"
        )
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*blk.fs.wind_to_grid[0]
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost - blk_pem.op_total_cost)
        # blk.pem_contract = Constraint(blk_pem.flowsheet().config.time,
        #                               rule=lambda b, t: m.contract_capacity <= blk_pem.outlet_state[0].flow_mol)

    m.wind_cap_cost = pyo.Param(default=1555, mutable=True)
    m.pem_cap_cost = pyo.Param(default=1630, mutable=True)

    n_weeks = 1
    # m.hydrogen_revenue = Expression(expr=m.h2_price_per_kg * m.contract_capacity / h2_mols_per_kg
    #                                     * 3600 * n_time_points)
    m.hydrogen_revenue = Expression(expr=sum([m.h2_price_per_kg * blk.fs.pem.outlet_state[0].flow_mol / h2_mols_per_kg
                                        * 3600 * n_time_points for blk in blks]))
    m.annual_revenue = Expression(expr=(sum([blk.profit for blk in blks]) + m.hydrogen_revenue) * 52 / n_weeks)
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

    for week in range(n_weeks):
        print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        opt.solve(m, tee=False)
        h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol) for i in range(n_time_points)])
        wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])
        wind_to_grid.append([pyo.value(blks[i].fs.wind_to_grid[0]) for i in range(n_time_points)])
        wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)])


    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()


    fig, ax1 = plt.subplots(figsize=(12, 8))

    # print(batt_in)
    # print(batt_out)
    # print(wind_out)

    # color = 'tab:green'
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('kW', )
    ax1.step(hours, wind_gen, label="Wind Generation")
    ax1.step(hours, wind_out, label="Wind to Grid")
    ax1.step(hours, h2_prod, label="H2 production")
    ax1.step(hours, wind_to_pem, label="Wind to Pem")
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
    # print(value(blks[0].fs.pem.system_capacity))
    print(value(m.pem_system_capacity))
    # print(value(m.contract_capacity))
    print(value(m.hydrogen_revenue))
    print(value(m.annual_revenue))
    print(value(m.NPV))


wind_pem_optimize()

# can't fix pem system capacity to anything above max production at lowest wind production
