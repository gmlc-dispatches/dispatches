import pyomo.environ as pyo
from idaes.apps.multiperiod.multiperiod import MultiPeriodModel
from RE_flowsheet import *
from load_LMP import *

design_opt = False


def wind_battery_pem_variable_pairs(m1, m2):
    """
    the power output and battery state are linked between time periods

        b1: current time block
        b2: next time block
    """
    pairs = [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
             (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput)]
    if design_opt:
        pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity),
                  (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power),
                  (m1.pem_system_capacity, m2.pem_system_capacity)]
    return pairs


def wind_battery_pem_periodic_variable_pairs(m1, m2):
    """
    the final power output and battery state must be the same as the intial power output and battery state

        b1: final time block
        b2: first time block
    """
    pairs = [(m1.fs.battery.state_of_charge[0], m2.fs.battery.initial_state_of_charge),
             (m1.fs.battery.energy_throughput[0], m2.fs.battery.initial_energy_throughput)]
    if design_opt:
        pairs += [(m1.fs.windpower.system_capacity, m2.fs.windpower.system_capacity),
                  (m1.fs.battery.nameplate_power, m2.fs.battery.nameplate_power),
                  (m1.pem_system_capacity, m2.pem_system_capacity)]
    return pairs


def wind_battery_pem_om_costs(m):
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


def wind_battery_pem_model(wind_resource_config):
    wind_mw = 200
    pem_bar = 8
    batt_mw = 100
    valve_cv = 0.0001
    tank_len_m = 0.1
    turb_p_lower_bound = 300
    turb_p_upper_bound = 450

    # m = create_model(wind_mw, pem_bar, batt_mw, valve_cv, tank_len_m)
    m = create_model(wind_mw, pem_bar, batt_mw, None, None, wind_resource_config=wind_resource_config)
    m.fs.windpower.system_capacity.unfix()
    m.fs.battery.nameplate_power.unfix()

    initialize_model(m, verbose=False)
    wind_battery_pem_om_costs(m)

    return m


def wind_battery_pem_mp_block(wind_resource_config):
    m = wind_battery_pem_model(wind_resource_config)
    battery_ramp_rate = 300
    batt = m.fs.battery

    batt.energy_down_ramp = pyo.Constraint(
        expr=batt.initial_state_of_charge - batt.state_of_charge[0] <= battery_ramp_rate)
    batt.energy_up_ramp = pyo.Constraint(
        expr=batt.state_of_charge[0] - batt.initial_state_of_charge <= battery_ramp_rate)
    return m


def wind_battery_pem_optimize():
    # create the multiperiod model object
    mp_battery_wind_pem = MultiPeriodModel(n_time_points=n_time_points,
                                      process_model_func=wind_battery_pem_model,
                                      linking_variable_func=wind_battery_pem_variable_pairs,
                                      periodic_variable_func=wind_battery_pem_periodic_variable_pairs)

    mp_battery_wind_pem.build_multi_period_model(wind_resource)

    m = mp_battery_wind_pem.pyomo_model
    blks = mp_battery_wind_pem.get_active_process_blocks()

    m.h2_price_per_kg = pyo.Param(default=h2_price_per_kg, mutable=True)
    m.pem_system_capacity = Var(domain=NonNegativeReals, initialize=20, units=pyunits.kW)
    # m.pem_system_capacity.fix(20)
    m.contract_capacity = Var(domain=NonNegativeReals, initialize=20, units=pyunits.mol/pyunits.second)

    # add market data for each block
    for blk in blks:
        blk_wind = blk.fs.windpower
        blk_battery = blk.fs.battery
        blk_pem = blk.fs.pem
        blk_pem.max_p = Constraint(blk_pem.flowsheet().config.time,
                                 rule=lambda b, t: b.electricity[t] <= m.pem_system_capacity)
        blk_pem.op_total_cost = Expression(
            expr=m.pem_system_capacity * blk_pem.op_cost / 8760 + blk_pem.var_cost * blk_pem.electricity[0],
            doc="total fixed cost of pem in $/hr"
        )
        blk.lmp_signal = pyo.Param(default=0, mutable=True)
        blk.revenue = blk.lmp_signal*(blk.fs.wind_to_grid[0] + blk_battery.elec_out[0])
        blk.profit = pyo.Expression(expr=blk.revenue - blk_wind.op_total_cost - blk_pem.op_total_cost)
        blk.pem_contract = Constraint(blk_pem.flowsheet().config.time,
                                      rule=lambda b, t: m.contract_capacity <= blk_pem.outlet_state[t].flow_mol)

    m.wind_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)
    m.pem_cap_cost = pyo.Param(default=pem_cap_cost, mutable=True)
    m.batt_cap_cost = pyo.Param(default=batt_cap_cost, mutable=True)

    n_weeks = 1
    m.hydrogen_revenue = Expression(expr=m.h2_price_per_kg * m.contract_capacity / h2_mols_per_kg
                                        * 3600 * n_time_points * n_time_points)
    # m.hydrogen_revenue = Expression(expr=sum([m.h2_price_per_kg * blk.fs.pem.outlet_state[0].flow_mol / h2_mols_per_kg
    #                                     * 3600 * n_time_points for blk in blks]))
    m.annual_revenue = Expression(expr=(sum([blk.profit for blk in blks]) + m.hydrogen_revenue) * 52 / n_weeks)
    m.NPV = Expression(expr=-(m.wind_cap_cost * blks[0].fs.windpower.system_capacity +
                              m.batt_cap_cost * blks[0].fs.battery.nameplate_power +
                              m.pem_cap_cost * m.pem_system_capacity) + PA * m.annual_revenue)
    m.obj = pyo.Objective(expr=-m.NPV)
    blks[0].fs.battery.initial_state_of_charge.fix(0)

    opt = pyo.SolverFactory('ipopt')
    opt.options['max_iter'] = 10000
    h2_prod = []
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []
    wind_to_batt = []
    batt_to_grid = []
    soc = []

    for week in range(n_weeks):
        print("Solving for week: ", week)
        for (i, blk) in enumerate(blks):
            blk.lmp_signal.set_value(weekly_prices[week][i])
        opt.solve(m, tee=False)
        h2_prod.append([pyo.value(blks[i].fs.pem.outlet_state[0].flow_mol * 3600) for i in range(n_time_points)])
        wind_gen.append([pyo.value(blks[i].fs.windpower.electricity[0]) for i in range(n_time_points)])
        wind_to_grid.append([pyo.value(blks[i].fs.wind_to_grid[0]) for i in range(n_time_points)])
        wind_to_pem.append([pyo.value(blks[i].fs.pem.electricity[0]) for i in range(n_time_points)])
        batt_to_grid.append([pyo.value(blks[i].fs.battery.elec_out[0]) for i in range(n_time_points)])
        wind_to_batt.append([pyo.value(blks[i].fs.battery.elec_in[0]) for i in range(n_time_points)])
        soc.append([pyo.value(blks[i].fs.battery.state_of_charge[0]) for i in range(n_time_points)])

    n_weeks_to_plot = 1
    hours = np.arange(n_time_points*n_weeks_to_plot)
    lmp_array = weekly_prices[0:n_weeks_to_plot].flatten()
    h2_prod = np.asarray(h2_prod[0:n_weeks_to_plot]).flatten()
    wind_to_pem = np.asarray(wind_to_pem[0:n_weeks_to_plot]).flatten()
    wind_gen = np.asarray(wind_gen[0:n_weeks_to_plot]).flatten()
    wind_out = np.asarray(wind_to_grid[0:n_weeks_to_plot]).flatten()
    batt_out = np.asarray(batt_to_grid[0:n_weeks_to_plot]).flatten()
    batt_in = np.asarray(wind_to_batt[0:n_weeks_to_plot]).flatten()
    batt_soc = np.asarray(soc[0:n_weeks_to_plot]).flatten()

    fig, ax1 = plt.subplots(2, 1, figsize=(12, 8))

    # color = 'tab:green'
    ax1[0].set_xlabel('Hour')
    ax1[0].set_ylabel('kW', )
    ax1[0].step(hours, wind_gen, label="Wind Generation")
    ax1[0].step(hours, wind_out, label="Wind to Grid")
    ax1[0].step(hours, batt_in, label="Wind to Batt")
    ax1[0].step(hours, batt_out, label="Batt to Grid")
    ax1[0].step(hours, batt_soc, label="Batt SOC")
    ax1[0].tick_params(axis='y', )
    ax1[0].legend()

    ax2 = ax1[0].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1[1].set_xlabel('Hour')
    ax1[1].set_ylabel('kg', )
    # ax1[1].step(hours, wind_gen, label="Wind Generation")
    # ax1[1].step(hours, wind_out, label="Wind to Grid")
    ax1[1].step(hours, h2_prod, label="H2 production")
    # ax1[1].step(hours, wind_to_pem, label="Wind to Pem")
    ax1[1].tick_params(axis='y', )
    ax1[1].legend()

    ax2 = ax1[1].twinx()
    color = 'k'
    ax2.set_ylabel('LMP [$/MWh]', color=color)
    ax2.plot(hours, lmp_array, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.show()

    print("wind mw", value(blks[0].fs.windpower.system_capacity))
    print("batt mw", value(blks[0].fs.battery.nameplate_power))
    print("pem mw", value(m.pem_system_capacity))
    print("h2 contract", value(m.contract_capacity))
    print("h2 rev", value(m.hydrogen_revenue))
    print("annual rev", value(m.annual_revenue))
    print("npv", value(m.NPV))


wind_battery_pem_optimize()

# at price 1.9$/kg H2, selling to grid is better
