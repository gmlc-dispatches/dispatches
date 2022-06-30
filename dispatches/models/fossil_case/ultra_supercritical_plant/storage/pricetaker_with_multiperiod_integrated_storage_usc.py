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

"""
This script uses the multiperiod model for the integrated ultra-supercritical
power plant with energy storage and performs market analysis using the
pricetaker assumption. The electricity prices, LMP (locational marginal prices)
are assumed to not change. The prices used in this study are either obtained
from a synthetic database.
"""

__author__ = "Naresh Susarla and Soraya Rawlings"


from pyomo.environ import Param, Objective, Expression, SolverFactory, value
import numpy as np
from dispatches.models.fossil_case.ultra_supercritical_plant.storage.\
    multiperiod_integrated_storage_usc import create_multiperiod_usc_model

# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)
scaling_obj = 1
scaling_factor = 1


def _get_lmp(number_hours):

    # Select lmp source data and scaling factor according to that
    use_rts_data = False
    use_mod_rts_data = True
    if use_rts_data:
        print('>>>>>> Using RTS lmp data')
        with open('rts_results_all_prices_base_case.npy', 'rb') as f:
            price = np.load(f)
    elif use_mod_rts_data:
        price = [22.9684, 21.1168, 20.4, 20.419,
                 20.419, 21.2877, 23.07, 25,
                 18.4634, 0, 0, 0,
                 0, 0, 0, 0,
                 19.0342, 23.07, 200, 200,
                 200, 200, 200, 200]
    else:
        print('>>>>>> Using NREL lmp data')
        price = np.load("nrel_scenario_average_hourly.npy")

    if use_rts_data:
        lmp = price[0:number_hours].tolist()
    elif use_mod_rts_data:
        lmp = price

    return lmp


def run_pricetaker_analysis(ndays, nweeks, tank_status, tank_min, tank_max):

    # Add number of days and hours per week
    number_hours = 24 * ndays
    n_time_points = nweeks * number_hours

    # Get LMP
    lmp = _get_lmp(number_hours)
    # Create the multiperiod model object. You can pass arguments to your
    # "process_model_func" for each time period using a dict of dicts as
    # shown here.  In this case, it is setting up empty dictionaries for
    # each time period.
    multiperiod_usc = create_multiperiod_usc_model(
        n_time_points=n_time_points, pmin=None, pmax=None
    )

    # Retrieve pyomo model and active process blocks (i.e. time blocks)
    m = multiperiod_usc.pyomo_model
    blks = multiperiod_usc.get_active_process_blocks()

    # Add lmp market data for each block
    count = 0
    for blk in blks:
        blk_usc_mp = blk.usc_mp
        blk.lmp_signal = Param(default=0, mutable=True)
        blk.revenue = lmp[count]*blk.usc_mp.fs.net_power * scaling_factor
        blk.operating_cost = Expression(
            expr=(
                (blk_usc_mp.fs.operating_cost
                 + blk_usc_mp.fs.plant_fixed_operating_cost
                 + blk_usc_mp.fs.plant_variable_operating_cost) / (365 * 24)
            ) * scaling_factor
        )
        blk.cost = Expression(expr=-(blk.revenue - blk.operating_cost))
        count += 1

    m.obj = Objective(expr=sum([blk.cost for blk in blks]) * scaling_obj)

    # Initial state for salt tank for different scenarios
    # Tank initial scenarios:"hot_empty","hot_full","hot_half_full"
    if tank_status == "hot_empty":
        blks[0].usc_mp.previous_salt_inventory_hot.fix(1103053.48)
        blks[0].usc_mp.previous_salt_inventory_cold.fix(tank_max-1103053.48)
    elif tank_status == "half_full":
        blks[0].usc_mp.previous_salt_inventory_hot.fix(tank_max/2)
        blks[0].usc_mp.previous_salt_inventory_cold.fix(tank_max/2)
    elif tank_status == "hot_full":
        blks[0].usc_mp.previous_salt_inventory_hot.fix(tank_max-tank_min)
        blks[0].usc_mp.previous_salt_inventory_cold.fix(tank_min)
    else:
        print("Unrecognized scenario! Try hot_empty, hot_full, or half_full")

    blks[0].usc_mp.previous_power.fix(447.66)

    # Plot results
    opt = SolverFactory('ipopt')
    hot_tank_level = []
    cold_tank_level = []
    net_power = []
    hxc_duty = []
    hxd_duty = []
    for week in range(nweeks):
        print()
        print(">>>>>> Solving for week {}: {} hours of operation in {} day(s) "
              .format(week + 1, number_hours, ndays))

        opt.solve(m, tee=True)

        hot_tank_level.append(
            [(value(blks[i].usc_mp.salt_inventory_hot) / scaling_factor)*1e-3
             for i in range(n_time_points)])
        cold_tank_level.append(
            [(value(blks[i].usc_mp.salt_inventory_cold) / scaling_factor)*1e-3
             for i in range(n_time_points)])
        net_power.append(
            [value(blks[i].usc_mp.fs.net_power)
             for i in range(n_time_points)])
        hxc_duty.append(
            [value(blks[i].usc_mp.fs.hxc.heat_duty[0])*1e-6
             for i in range(n_time_points)])
        hxd_duty.append(
            [value(blks[i].usc_mp.fs.hxd.heat_duty[0])*1e-6
             for i in range(n_time_points)])

    return (lmp, m, blks, hot_tank_level, cold_tank_level,
            net_power, hxc_duty, hxd_duty)


def print_results(m, blks):
    c = 0
    print('Objective: {:.4f}'.format(value(m.obj)))
    for blk in blks:
        print()
        print('Period {}'.format(c+1))
        print(' Net power: {:.4f}'.format(
            value(blks[c].usc_mp.fs.net_power)))
        print(' Plant Power Out: {:.4f}'.format(
            value(blks[c].usc_mp.fs.plant_power_out[0])))
        print(' ES Turbine Power: {:.4f}'.format(
            value(blks[c].usc_mp.fs.es_turbine.work_mechanical[0])*(-1e-6)))
        print(' Plant heat duty: {:.4f}'.format(
            value(blks[c].usc_mp.fs.plant_heat_duty[0])))
        print(' Cost ($): {:.4f}'.format(value(blks[c].cost) / scaling_factor))
        print(' Revenue ($): {:.4f}'
              .format(value(blks[c].revenue) / scaling_factor))
        print(' Operating cost ($): {:.4f}'
              .format(value(blks[c].operating_cost) / scaling_factor))
        print(' Specific Operating cost ($/MWh): {:.4f}'.format(
            (value(blks[c].operating_cost) / scaling_factor) /
            value(blks[c].usc_mp.fs.net_power)))
        print(' Cycle efficiency (%): {:.4f}'.format(
            value(blks[c].usc_mp.fs.cycle_efficiency)))
        print(' Boiler efficiency (%): {:.4f}'.format(
            value(blks[c].usc_mp.fs.boiler_eff) * 100))
        print(' Boiler heat duty: {:.4f}'.format(
            value(blks[c].usc_mp.fs.boiler.heat_duty[0]) * 1e-6))
        print(' Boiler flow mol (mol/s): {:.4f}'.format(
            value(blks[c].usc_mp.fs.boiler.outlet.flow_mol[0])))
        print(' Previous salt inventory (mton): {:.4f}'.format(
            (value(blks[c].usc_mp.previous_salt_inventory_hot) /
             scaling_factor) * 1e-3))
        print(' Salt from HXC (mton): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxc.outlet_2.flow_mass[0]) * 3600 * 1e-3))
        print(' Salt from HXD (mton): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxd.outlet_1.flow_mass[0]) * 3600 * 1e-3))
        print(' HXC Duty (MW): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxc.heat_duty[0]) * 1e-6))
        print(' HXD Duty (MW): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxd.heat_duty[0]) * 1e-6))
        print(' Split fraction to HXC: {:.4f}'.format(
            value(blks[c].usc_mp.fs.ess_hp_split.split_fraction[0, "to_hxc"])))
        print(' Split fraction to HXD: {:.4f}'.format(
            value(blks[c].usc_mp.fs.ess_bfp_split.split_fraction[0, "to_hxd"])))
        print(' Salt flow HXC (kg/s): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxc.outlet_2.flow_mass[0])))
        print(' Salt flow HXD (kg/s): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxd.outlet_1.flow_mass[0])))
        print(' Steam flow HXC (mol/s): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxc.outlet_1.flow_mol[0])))
        print(' Steam flow HXD (mol/s): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxd.outlet_2.flow_mol[0])))
        print(' Delta T in HXC (kg): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxc.delta_temperature_in[0])))
        print(' Delta T out HXC (kg): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxc.delta_temperature_out[0])))
        print(' Delta T in HXD (kg): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxd.delta_temperature_in[0])))
        print(' Delta T out HXD (kg): {:.4f}'.format(
            value(blks[c].usc_mp.fs.hxd.delta_temperature_out[0])))
        c += 1


def plot_results(ndays, nweeks, lmp, m, blks, hot_tank_level, cold_tank_level,
                 net_power, hxc_duty, hxd_duty, tank_max):
    max_power = 436  # in MW
    max_power_storage = 30  # in MW
    max_power_total = max_power + max_power_storage
    min_storage_heat_duty = 10  # in MW
    max_storage_heat_duty = 200  # in MW

    n_time_points = 24 * ndays
    hours = np.arange(n_time_points * nweeks)
    lmp_array = np.asarray(lmp[0:n_time_points])
    hot_tank_array = np.asarray(hot_tank_level[0:nweeks]).flatten()
    cold_tank_array = np.asarray(cold_tank_level[0:nweeks]).flatten()

    # Convert array to list to include hot tank level at time zero
    hot_tank_array0 = (value(blks[0].usc_mp.previous_salt_inventory_hot) /
                       scaling_factor) * 1e-3
    cold_tank_array0 = (value(blks[0].usc_mp.previous_salt_inventory_cold) /
                        scaling_factor) * 1e-3
    hours_list = hours.tolist() + [n_time_points]
    hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
    cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()

    font = {'size': 16}
    plt.rc('font', **font)
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    color = ['r', 'b', 'tab:green', 'k', 'tab:orange']
    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Tank Level (metric ton)',
                   color=color[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.axhline((tank_max / scaling_factor)*1e-3, ls=':', lw=1.75,
                color=color[4])
    plt.text(n_time_points / 2 - 1.5,
             (tank_max / scaling_factor) * 1e-3 + 100, 'max salt',
             color=color[4])
    ax1.step(  # [x + 1 for x in hours], hot_tank_array,
        hours_list, hot_tank_list,
        marker='^', ms=4, label='Hot Salt',
        lw=1, color=color[0])
    ax1.step(  # [x + 1 for x in hours], hot_tank_array,
        hours_list, cold_tank_list,
        marker='v', ms=4, label='Cold Salt',
        lw=1, color=color[1])
    ax1.legend(loc="center right", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=2))

    ax2 = ax1.twinx()
    ax2.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax2.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', lw=1,
             color=color[2])
    ax2.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('salt_tank_level_24h.png')

    font = {'size': 18}
    plt.rc('font', **font)

    power_array = np.asarray(net_power[0:nweeks]).flatten()
    # Convert array to list to include net power at time zero
    power_array0 = value(blks[0].usc_mp.previous_power)
    power_list = [power_array0] + power_array.tolist()

    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Net Power Output (MW)',
                   color=color[1])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.text(n_time_points / 2 - 3, max_power - 5.5, 'max plant power',
             color=color[4])
    plt.text(n_time_points / 2 - 2.8, max_power_total + 1, 'max net power',
             color=color[4])
    plt.axhline(max_power, ls='-.', lw=1.75,
                color=color[4])
    plt.axhline(max_power_total, ls=':', lw=1.75,
                color=color[4])
    ax3.step(hours_list, power_list,
             marker='o', ms=4,
             lw=1, color=color[1])
    ax3.tick_params(axis='y',
                    labelcolor=color[1])
    ax3.set_xticks(np.arange(0, n_time_points * nweeks + 1, step=2))

    ax4 = ax3.twinx()
    ax4.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax4.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', lw=1,
             color=color[2])
    ax4.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('plant_power_24h.png')

    zero_point = True
    hxc_array = np.asarray(hxc_duty[0:nweeks]).flatten()
    hxd_array = np.asarray(hxd_duty[0:nweeks]).flatten()
    hxc_duty0 = 0  # zero since the plant is not operating
    hxc_duty_list = [hxc_duty0] + hxc_array.tolist()
    hxd_duty0 = 0  # zero since the plant is not operating
    hxd_duty_list = [hxd_duty0] + hxd_array.tolist()

    fig3, ax5 = plt.subplots(figsize=(12, 8))
    ax5.set_xlabel('Time Period (hr)')
    ax5.set_ylabel('Storage Heat Duty (MW)',
                   color=color[3])
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.grid(linestyle=':', which='both',
             color='gray', alpha=0.30)
    plt.text(n_time_points / 2 - 2.2, max_storage_heat_duty + 1, 'max storage',
             color=color[4])
    plt.text(n_time_points / 2 - 2, min_storage_heat_duty - 6.5, 'min storage',
             color=color[4])
    plt.axhline(max_storage_heat_duty, ls=':', lw=1.75,
                color=color[4])
    plt.axhline(min_storage_heat_duty, ls=':', lw=1.75,
                color=color[4])
    if zero_point:
        ax5.step(hours_list, hxc_duty_list,
                 marker='^', ms=4, label='Charge',
                 color=color[0])
        ax5.step(hours_list, hxd_duty_list,
                 marker='v', ms=4, label='Discharge',
                 color=color[1])
    else:
        ax5.step([x + 1 for x in hours], hxc_array,
                 marker='^', ms=4, lw=1,
                 label='Charge',
                 color=color[0])
        ax5.step([x + 1 for x in hours], hxd_array,
                 marker='v', ms=4, lw=1,
                 label='Discharge',
                 color=color[1])
    ax5.legend(loc="center right", frameon=False)
    ax5.tick_params(axis='y',
                    labelcolor=color[3])
    ax5.set_xticks(np.arange(0, n_time_points * nweeks + 1, step=2))

    ax6 = ax5.twinx()
    ax6.set_ylabel('LMP ($/MWh)',
                   color=color[2])
    ax6.step([x + 1 for x in hours], lmp_array,
             marker='o', ms=3, alpha=0.5,
             ls='-', color=color[2])
    ax6.tick_params(axis='y',
                    labelcolor=color[2])
    plt.savefig('heat_exchanger_duties_24h.png')

    plt.show()


if __name__ == "__main__":

    ndays = 1
    nweeks = 1

    tank_status = "hot_empty"
    tank_min = 1  # in kg
    tank_max = 6739292  # in kg

    (lmp, m, blks, hot_tank_level, cold_tank_level, net_power,
     hxc_duty, hxd_duty) = run_pricetaker_analysis(ndays, nweeks, tank_status,
                                                   tank_min, tank_max)

    print_results(m, blks)

    plot_results(ndays, nweeks, lmp, m, blks, hot_tank_level,
                 cold_tank_level, net_power, hxc_duty, hxd_duty, tank_max)
