##############################################################################
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
#
##############################################################################

"""This script uses the multiperiod model in the GDP integrated
ultra-supercritical power plant model with energy storage and performs
market analysis using the pricetaker assumption. The electricity
prices or LMP (locational marginal prices) are assumed to not
change. The prices used in this study are obtained from a synthetic
database.

"""

__author__ = "Soraya Rawlings"

import csv
import json
import os
import copy
import numpy as np
import logging

import pyomo.environ as pyo
from pyomo.environ import (Constraint, Expression,
                           Var, Objective,
                           SolverFactory,
                           value, RangeSet, maximize)
from pyomo.contrib.fbbt.fbbt import _prop_bnds_root_to_leaf_map
from pyomo.core.expr.numeric_expr import ExternalFunctionExpression
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)

from idaes.core.solvers.get_solver import get_solver

from gdp_multiperiod_usc_pricetaker_unfixed_area import create_gdp_multiperiod_usc_model

# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
font = {'size':16}
plt.rc('axes', titlesize=24)
plt.rc('font', **font)


def _get_lmp(hours_per_day=None, nhours=None):

    # Select lmp source data and scaling factor according to that
    use_rts_data = False
    if use_rts_data:
        use_mod_rts_data = False
    else:
        use_mod_rts_data = True

    nhigh_lmp = 0
    nlow_lmp = 0
    if use_rts_data:
        print('>> Using RTS lmp data')
        with open('rts_results_all_prices_base_case.npy', 'rb') as f:
            dispatch = np.load(f)
            price = np.load(f)
        lmp = price[0:nhours].tolist()
    elif use_mod_rts_data:
        # price = [22.9684, 21.1168, 20.4, 20.419, 0, 0, 200, 200]
        price = [
            # 22.9684, 21.1168, 20.4, 20.419,
            52.9684, 21.1168, 10.4, 5.419,
            # 20.419, 21.2877, 23.07, 25,
            # 18.4634, 0, 0, 0,
            0, 0, 0, 0,
            # 19.0342, 23.07, 200, 200,
            200, 200, 200, 200,
        ]
        max_lmp = max(price)
        min_lmp = min(price)
        for i in price:
            if i >= (max_lmp - 5):
                nhigh_lmp += 1
            elif  i <= (min_lmp + 5):
                nlow_lmp += 1
        print('** {} lmp prices ($/MWh) between [max_lmp, max_lmp - 5]: [{}, {}]'.format(
            nhigh_lmp, max_lmp, max_lmp - 5))
        print('   {} lmp prices ($/MWh) between [min_lmp, min_lmp + 5]: [{}, {}]'.format(
            nlow_lmp, min_lmp, min_lmp + 5))
        print()

        if len(price) < hours_per_day:
            print()
            print('**ERROR: I need more LMP data!')
            raise Exception
        lmp = price
    else:
        print('>> Using NREL lmp data')
        price = np.load("nrel_scenario_average_hourly.npy")
        # print(lmp)

    return lmp, nhigh_lmp, nlow_lmp


def print_model(solver_obj,
                mdl,
                mdl_data,
                csvfile,
                lmp=None,
                nweeks=None,
                nhours=None,
                n_time_points=None):

    # m_iter = mdl_data.master_iteration
    m_iter = solver_obj.iteration

    mdl.disjunction1_selection = {}
    hot_tank_level_iter = []
    cold_tank_level_iter = []
    boiler_heat_duty_iter = []
    hxc_duty_iter = []
    hxd_duty_iter = []
   
    print('       ___________________________________________')
    print('        Schedule')
    print('         Obj ($): {:.4f}'.format(
        (value(mdl.obj) / scaling_cost) / scaling_obj))
    print('         Cycles: {} charge, {} discharge, {} no storage'.format(
        sum(mdl.blocks[blk].process.usc.fs.charge_mode_disjunct.binary_indicator_var.value
            for blk in mdl.blocks),
        sum(mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct.binary_indicator_var.value
            for blk in mdl.blocks),
        sum(mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var.value
            for blk in mdl.blocks)))

    for blk in mdl.blocks:
        blk_process_charge = mdl.blocks[blk].process.usc.fs.charge_mode_disjunct
        blk_process_discharge = mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct
        blk_process_no_storage = mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct
        if blk_process_charge.binary_indicator_var.value == 1:
            print('         Period {}: Charge (HXC: {:.2f} MW, {:.2f} m2)'.format(
                blk,
                value(blk_process_charge.hxc.heat_duty[0]) * 1e-6,
                value(blk_process_charge.hxc.area)))
        if blk_process_discharge.binary_indicator_var.value == 1:
            print('         Period {}: Discharge (HXD: {:.2f} MW, {:.2f} m2)'.format(
                blk,
                value(blk_process_discharge.hxd.heat_duty[0]) * 1e-6,
                value(blk_process_discharge.hxd.area)))
        if blk_process_no_storage.binary_indicator_var.value == 1:
            print('         Period {}: No storage'.format(blk))

    print()
    for blk in mdl.blocks:
        blk_process = mdl.blocks[blk].process.usc
        blk_process_charge = mdl.blocks[blk].process.usc.fs.charge_mode_disjunct
        blk_process_discharge = mdl.blocks[blk].process.usc.fs.discharge_mode_disjunct
        blk_process_no_storage = mdl.blocks[blk].process.usc.fs.no_storage_mode_disjunct
        print('       Time period {} '.format(blk+1))
        print('        Charge: {}'.format(
            blk_process_charge.binary_indicator_var.value))
        print('        Discharge: {}'.format(
            blk_process_discharge.binary_indicator_var.value))
        print('        No storage: {}'.format(
            blk_process_no_storage.binary_indicator_var.value))
        if blk_process_charge.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[m_iter] = 'Charge'
            print('         HXC area (m2): {:.4f}'.format(
                value(blk_process_charge.hxc.area)))
            print('         HXC Duty (MW): {:.4f}'.format(
                value(blk_process_charge.hxc.heat_duty[0]) * 1e-6))
            print('         HXC Delta temperature in/out (K): {:.4f}/{:.4f}'.format(
                value(blk_process_charge.hxc.delta_temperature_in[0]),
                value(blk_process_charge.hxc.delta_temperature_out[0])))
            print('         HXC salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_charge.hxc.tube_inlet.temperature[0]),
                value(blk_process_charge.hxc.tube_outlet.temperature[0])))
            print('         Salt flow HXC (kg/s): {:.4f}'.format(
                value(blk_process_charge.hxc.tube_outlet.flow_mass[0])))
            print('         HXC steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_charge.hxc.hot_side.properties_in[0].temperature),
                value(blk_process_charge.hxc.hot_side.properties_out[0].temperature)
            ))
            print('         Steam flow HXC (mol/s): {:.4f}'.format(
                value(blk_process_charge.hxc.shell_outlet.flow_mol[0])))
            if not new_design:
                print('         Cooling heat duty (MW): {:.4f}'.format(
                    value(blk_process_charge.cooler.heat_duty[0]) * 1e-6))
        elif blk_process_discharge.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[m_iter] = 'Discharge'
            print('         HXD area (m2): {:.4f}'.format(
                value(blk_process_discharge.hxd.area)))
            print('         HXD Duty (MW): {:.4f}'.format(
                value(blk_process_discharge.hxd.heat_duty[0]) * 1e-6))
            print('         HXD Delta temperature in/out (K): {:.4f}/{:.4f}'.format(
                value(blk_process_discharge.hxd.delta_temperature_in[0]),
                value(blk_process_discharge.hxd.delta_temperature_out[0])))
            print('         HXD salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_discharge.hxd.shell_inlet.temperature[0]),
                value(blk_process_discharge.hxd.shell_outlet.temperature[0])))
            print('         Salt flow HXD (kg/s): {:.4f}'.format(
                value(blk_process_discharge.hxd.shell_outlet.flow_mass[0])))
            print('         HXD steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(blk_process_discharge.hxd.cold_side.properties_in[0].temperature),
                value(blk_process_discharge.hxd.cold_side.properties_out[0].temperature)
            ))
            print('         Steam flow HXD (mol/s): {:.4f}'.format(
                value(blk_process_discharge.hxd.tube_outlet.flow_mol[0])))
            print('         ES turbine work (MW): {:.4f}'.format(
                value(blk_process_discharge.es_turbine.work_mechanical[0]) * -1e-6))
        elif blk_process_no_storage.binary_indicator_var.value == 1:
            mdl.disjunction1_selection[m_iter] = 'No_storage'
            print('         Salt flow (kg/s): {:.4f}'.format(
                value(blk_process.fs.salt_storage)))
        else:
            print('        No other operation mode is available!')
        print('        Net power: {:.4f}'.format(value(blk_process.fs.net_power)))
        print('        Coal heat duty: {:.4f}'.format(value(blk_process.fs.coal_heat_duty)))
        print('        Discharge turbine work (MW): {:.4f}'.format(
            value(blk_process.fs.discharge_turbine_work)))
        if not new_design:
            print('        Cooler heat duty: {:.4f}'.format(
                value(blk_process.fs.cooler_heat_duty)))
        print('        Efficiencies (%): boiler: {:.4f}, cycle: {:.4f}'.format(
            value(blk_process.fs.boiler_efficiency) * 100,
            value(blk_process.fs.cycle_efficiency) * 100))
        print('        Boiler heat duty: {:.4f}'.format(
            value(blk_process.fs.boiler.heat_duty[0]) * 1e-6))
        print('        Boiler flow mol (mol/s): {:.4f}'.format(
            value(blk_process.fs.boiler.outlet.flow_mol[0])))
        print('        Salt to storage (kg/s) [mton]: {:.4f} [{:.4f}]'.format(
            value(blk_process.fs.salt_storage),
            value(blk_process.fs.salt_storage) * 3600 * factor_mton))
        print('        Hot salt inventory (mton): {:.4f}, previous: {:.4f}'.format(
            value(blk_process.salt_inventory_hot),
            value(blk_process.previous_salt_inventory_hot)))
        print('        Makeup water flow (mol/s): {:.4f}'.format(
            value(blk_process.fs.condenser_mix.makeup.flow_mol[0])))
        # print('        Revenue (M$/year): {:.4f}'.format(
        #     value(mdl.blocks[blk].process.revenue)))
        # print('        Total op cost ($/h): {:.4f}'.format(
        #     value(mdl.blocks[blk].process.total_cost)))
        print('        Storage cost ($/h): {:.4f}'.format(
            value(blk_process.fs.storage_capital_cost)))
        print('        Fuel cost ($/h): {:.4f}'.format(
            value(blk_process.fs.fuel_cost)))
        print('        Plant fixed op cost ($/h): {:.4f}'.format(
            value(blk_process.fs.plant_fixed_operating_cost)))
        print('        Plant variable op cost ($/h): {:.4f}'.format(
            value(blk_process.fs.plant_variable_operating_cost)))
        print()

        # Save data for each NLP subproblem and plot results
        mdl.objective_val = {}
        mdl.boiler_heat_duty_val = {}
        mdl.discharge_turbine_work_val = {}
        mdl.hxc_area_val = {}
        mdl.hxd_area_val = {}
        mdl.hot_salt_temp_val = {}
        mdl.objective_val[m_iter] = (value(mdl.obj) / scaling_cost) / scaling_obj
        mdl.period = blk
        mdl.boiler_heat_duty_val[m_iter] = 1e-6 * value(blk_process.fs.boiler.heat_duty[0])
        mdl.discharge_turbine_work_val[m_iter] = value(blk_process.fs.discharge_turbine_work)
        mdl.hxc_area_val[m_iter] = value(blk_process_charge.hxc.area)
        mdl.hxd_area_val[m_iter] = value(blk_process_discharge.hxd.area)
        mdl.hot_salt_temp_val[m_iter] = value(blk_process_charge.hxc.tube_outlet.temperature[0])

        if save_results:
            writer = csv.writer(csvfile)
            writer.writerow(
                (m_iter,
                 mdl.period,
                 mdl.disjunction1_selection[m_iter],
                 mdl.boiler_heat_duty_val[m_iter],
                 mdl.discharge_turbine_work_val[m_iter],
                 mdl.hxc_area_val[m_iter],
                 mdl.hxd_area_val[m_iter],
                 mdl.hot_salt_temp_val[m_iter],
                 mdl.objective_val[m_iter])
            )
            csvfile.flush()

    print('       ___________________________________________')

    hot_tank_level_iter.append(
        [(pyo.value(mdl.blocks[i].process.usc.salt_inventory_hot)) # in mton
         for i in range(n_time_points)])
    cold_tank_level_iter.append(
        [(pyo.value(mdl.blocks[i].process.usc.salt_inventory_cold)) # in mton
         for i in range(n_time_points)])
    boiler_heat_duty_iter.append([pyo.value(mdl.blocks[i].process.usc.fs.boiler.heat_duty[0]) * 1e-6
                                  for i in range(n_time_points)])
    hxc_duty_iter.append([pyo.value(mdl.blocks[i].process.usc.fs.charge_mode_disjunct.indicator_var) *
                          pyo.value(mdl.blocks[i].process.usc.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6
                          for i in range(n_time_points)])
    hxd_duty_iter.append([pyo.value(mdl.blocks[i].process.usc.fs.discharge_mode_disjunct.indicator_var) *
                          pyo.value(mdl.blocks[i].process.usc.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6
                          for i in range(n_time_points)])

    # Save list of colors to be used in plots
    c = ['darkred', 'midnightblue', 'tab:green', 'k', 'gray']

    # Save and convert array to list to include values at time zero
    hours = np.arange(n_time_points * nweeks)
    lmp_array = np.asarray(lmp[0:n_time_points])
    hot_tank_array = np.asarray(hot_tank_level_iter[0:nweeks]).flatten()
    cold_tank_array = np.asarray(cold_tank_level_iter[0:nweeks]).flatten()
    hot_tank_array0 = value(mdl.blocks[0].process.usc.previous_salt_inventory_hot)
    cold_tank_array0 = value(mdl.blocks[0].process.usc.previous_salt_inventory_cold)
    hours_list = hours.tolist() + [nhours]
    hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
    cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()
    hxc_array = np.asarray(hxc_duty_iter[0:nweeks]).flatten()
    hxd_array = np.asarray(hxd_duty_iter[0:nweeks]).flatten()
    hxc_duty_list = [0] + hxc_array.tolist()
    hxd_duty_list = [0] + hxd_array.tolist()
    boiler_heat_duty_array = np.asarray(boiler_heat_duty_iter[0:nweeks]).flatten()
    boiler_heat_duty_list = [0] + boiler_heat_duty_array.tolist()

    # Plot salt tank profiles and heat duty of boiler and storage heat
    # exchangers at each master iteration.
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Amount (metric ton)', color=c[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    # ax1.set_ylim((0, 7000))
    plt.axhline(tank_max, ls=':', lw=1.5, color=c[4])
    ax1.step(hours_list, hot_tank_list, marker='o', ms=8, lw=1.5, color=c[0], alpha=0.85,
             label='Hot Tank')
    ax1.fill_between(hours_list, hot_tank_list, step="pre", color=c[0], alpha=0.35)
    ax1.step(hours_list, cold_tank_list, marker='o', ms=8, lw=1.5, color=c[1], alpha=0.65,
             label='Cold Tank')
    ax1.fill_between(hours_list, cold_tank_list, step="pre", color=c[1], alpha=0.10)
    ax1.legend(loc="upper left", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=1))
    ax2 = ax1.twinx()
    ax2.set_ylim((-25, 225))
    ax2.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax2.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.5, ls='-', lw=1.5, color=c[2])
    ax2.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/salt_tank_level_master_iter{}.png'.format(
        nhours, m_iter))
    plt.close(fig1)

    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Heat Duty (MW)', color=c[3])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    ax3.set_ylim((-25, 825))
    ax3.step(hours_list, boiler_heat_duty_list, marker='o', ms=8, ls='-', lw=1.5, alpha=0.55, color=c[3],
             label='Boiler')
    ax3.fill_between(hours_list, boiler_heat_duty_list, step="pre", color=c[3], alpha=0.15)
    plt.axhline(max_storage_heat_duty, ls=':', lw=1.5, color=c[4])
    plt.axhline(min_storage_heat_duty, ls=':', lw=1.5, color=c[4])
    ax3.step(hours_list, hxc_duty_list, marker='o', ms=8, color=c[0], alpha=0.75,
             label='Charge')
    ax3.fill_between(hours_list, hxc_duty_list, step="pre", color=c[0], alpha=0.25)
    ax3.step(hours_list, hxd_duty_list, marker='o', ms=8, color=c[1], alpha=0.75,
             label='Discharge')
    ax3.fill_between(hours_list, hxd_duty_list, step="pre", color=c[1], alpha=0.25)
    ax3.legend(loc="center right", frameon=False)
    ax3.tick_params(axis='y')
    ax3.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=1))
    ax4 = ax3.twinx()
    ax4.set_ylim((-25, 225))
    ax4.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax4.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax4.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/heat_duty_master_iter{}.png'.format(
        nhours, m_iter))
    plt.close(fig2)

    log_close_to_bounds(mdl)
    log_infeasible_constraints(mdl)


def create_csv_header(nhours):

    csvfile = open('results/gdp_mp_unfixed_area_{}h/results_subnlps_master_iter.csv'.format(nhours),
                   'w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(
        ('Iteration',
         'TimePeriod(hr)',
         'OperationMode',
         'BoilerHeatDuty(MW)',
         'DischargeWork(MW)',
         'HXCArea',
         'HXDArea',
         'SaltHotTemp',
         'Obj($/hr)')
    )
    return csvfile


def run_pricetaker_analysis(hours_per_day=None,
                            nhours=None,
                            ndays=None,
                            nweeks=None,
                            n_time_points=None,
                            pmin=None,
                            tank_status=None,
                            tank_min=None,
                            tank_max=None):

    # Get LMP data
    lmp, nhigh_lmp, nlow_lmp = _get_lmp(hours_per_day=hours_per_day, nhours=nhours)

    # Create the multiperiod model object. You can pass arguments to
    # the "process_model_func" for each time period using a dict of
    # dicts as shown here.  In this case, it is setting up empty
    # dictionaries for each time period.
    gdp_multiperiod_usc = create_gdp_multiperiod_usc_model(
        n_time_points=n_time_points,
        pmin=pmin,
        pmax=None
    )

    # Retrieve pyomo model and active process blocks
    m = gdp_multiperiod_usc.pyomo_model
    blks = gdp_multiperiod_usc.get_active_process_blocks()

    ##################################################################
    # Add nonanticipativity constraints
    ##################################################################
    m.hours_set = RangeSet(0, nhours - 1)
    m.hours_set2 = RangeSet(0, nhours - 2)

    # Add constraint to save calculate charge and discharge area in a
    # global variable
    @m.Constraint(m.hours_set2)
    def constraint_charge_previous_area(b, h):
        return (
            b.blocks[h + 1].process.usc.fs.charge_area ==
            b.blocks[h].process.usc.fs.charge_area
        )

    @m.Constraint(m.hours_set2)
    def constraint_discharge_previous_area(b, h):
        return (
            b.blocks[h + 1].process.usc.fs.discharge_area ==
            b.blocks[h].process.usc.fs.discharge_area
        )

    # @m.Constraint(m.hours_set)
    # def constraint_charge_area_lb(b, h):
    #     return (
    #         b.blocks[h].process.usc.fs.charge_area >= 1000
    #     )
    # @m.Constraint(m.hours_set)
    # def constraint_discharge_area_lb(b, h):
    #     return (
    #         b.blocks[h].process.usc.fs.discharge_area >= 1000
    #     )

    # Declare constraint to ensure that the discharge heat exchanger
    # has the same temperature for the hot salt than the one obtained
    # during charge cycle.
    @m.Constraint(m.hours_set)
    def constraint_discharge_hot_salt_temperature(b, h):
        return (
            b.blocks[h].process.usc.fs.discharge_mode_disjunct.hxd.shell_inlet.temperature[0] ==
            b.blocks[h].process.usc.fs.hot_salt_temp
        )

    ##################################################################
    # Add logical constraints
    ##################################################################
    discharge_min_salt = 379 # in mton, 8MW min es turbine
    # discharge_min_salt = 1 # in mton, 8MW min es turbine
    min_hot_salt = 2000
    @m.Constraint(m.hours_set)
    def _constraint_no_discharge_with_min_hot_tank(b, h):
        if h <= 2:
            a = min_hot_salt
        else:
            a = discharge_min_salt
        return (
            (b.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var * a) <=
            blks[h].usc.previous_salt_inventory_hot
        )

    # Add a minimum number of charge, discharge, and no storage
    # operation modes. Note: For charge, the minimum number of cycles
    # is based on the number of low lmp values obtained above, while
    # for discharge, the minimum number of cycles is based on the
    # number of high lmp values.
    @m.Constraint()
    def _constraint_min_charge(b):
        return sum(b.blocks[h].process.usc.fs.charge_mode_disjunct.binary_indicator_var
                   for h in b.hours_set) >= nlow_lmp
    @m.Constraint()
    def _constraint_min_discharge(b):
        return sum(b.blocks[h].process.usc.fs.discharge_mode_disjunct.binary_indicator_var
                   for h in b.hours_set) >= nhigh_lmp - 1
    # @m.Constraint()
    # def _constraint_min_no_storage(b):
    #     return sum(b.blocks[h].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var
    #                for h in b.hours_set) >= 1


    # if tank_status == "hot_empty":
    #     # Add logical constraint to help reduce the alternatives to explore
    #     # when periodic behavior is expected
    #     @m.Constraint()
    #     def _logic_constraint_no_discharge_time0(b):
    #         return b.blocks[0].process.usc.fs.discharge_mode_disjunct.binary_indicator_var == 0
    #     @m.Constraint()
    #     def _logic_constraint_no_charge_at_timen(b):
    #         return (
    #             (b.blocks[0].process.usc.fs.charge_mode_disjunct.binary_indicator_var
    #              + b.blocks[nhours - 1].process.usc.fs.charge_mode_disjunct.binary_indicator_var) <= 1
    #         )
    #     # @m.Constraint()
    #     # def _logic_constraint_no_storage_time0_no_charge_at_timen(m):
    #     #     return (
    #     #         (m.blocks[0].process.usc.fs.no_storage_mode_disjunct.binary_indicator_var
    #     #          + m.blocks[nhours - 1].process.usc.fs.charge_mode_disjunct.binary_indicator_var) <= 1
    #     #     )
    # elif tank_status == "hot_full":
    #     @m.Constraint()
    #     def _logic_constraint_no_discharge_at_timen(m):
    #         return (
    #             (m.blocks[0].process.usc.fs.discharge_mode_disjunct.binary_indicator_var
    #              + m.blocks[nhours - 1].process.usc.fs.discharge_mode_disjunct.binary_indicator_var) <= 1
    #         )

    # Add lmp market data for each block
    count = 0
    for blk in blks:
        blk.revenue = pyo.Expression(
            expr=(lmp[count] * blk.usc.fs.net_power)
        )

        # # Add expression to calculate total operating costs. Note that
        # # these costs are scaled using a scaling cost factor
        # blk.total_cost = pyo.Expression(
        #     expr=(
        #         blk.usc.fs.fuel_cost +
        #         blk.usc.fs.plant_fixed_operating_cost +
        #         blk.usc.fs.plant_variable_operating_cost +
        #         blk.usc.fs.storage_capital_cost
        #     )
        # )

        # # Declare expression to calculate the total profit. All the
        # # costs are in $ per hour
        # blk.profit = pyo.Expression(
        #     expr=(
        #         blk.revenue -
        #         blk.total_cost
        #         # (lmp[count] * blk.usc.fs.net_power) -
        #         # (
        #         #     blk.usc.fs.fuel_cost +
        #         #     blk.usc.fs.plant_fixed_operating_cost +
        #         #     blk.usc.fs.plant_variable_operating_cost +
        #         #     blk.usc.fs.storage_capital_cost
        #         # )
        #     ) * scaling_cost
        # )
        count += 1

    # m.obj = pyo.Objective(
    #     expr=sum([blk.profit for blk in blks]) * scaling_obj,
    #     sense=maximize
    # )
    m.obj = pyo.Objective(
        expr=sum(
            [blk.revenue -
             (blk.usc.fs.fuel_cost +
              blk.usc.fs.plant_fixed_operating_cost +
              blk.usc.fs.plant_variable_operating_cost) -
             blk.usc.fs.storage_capital_cost
             for blk in blks]
        ) * scaling_obj,
        sense=maximize
    )

    # Initial state for linking variables: power and salt
    # tank. Different tank scenarios are included for the Solar salt
    # tank levels and the previous tank level of the tank is based on
    # that.
    if tank_status == "hot_empty":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_min)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_max-tank_min)
    elif tank_status == "hot_half_full":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_max/2)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_max/2)
    elif tank_status == "hot_full":
        blks[0].usc.previous_salt_inventory_hot.fix(tank_max-tank_min)
        blks[0].usc.previous_salt_inventory_cold.fix(tank_min)
    else:
        print("Unrecognized scenario! Try hot_empty, hot_full, or hot_half_full")

    blks[0].usc.previous_power.fix(447.66)

    # Initialize disjunctions
    print()
    print()
    print('>>Initializing disjuncts')
    if tank_status == "hot_empty":
        for k in range(nhours):
            # if k <= (nhours / 3) - 1:
            #     blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
            #     blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
            #     blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(1)
            # elif k <= 2 * (nhours / 3) - 1:
            #     blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(1)
            #     blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
            #     blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
            if k <= (nhours / 2) - 1:
                blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(1)
                blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
                blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
            # if k >= (nhours / 2) - 1:
            #     blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(1)
            #     blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
            #     blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
            else:
                blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
                blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(1)
                blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
    elif tank_status == "hot_full":
        blks[0].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
        blks[0].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(1)
        blks[0].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
        for k in range(nhours):
            if k >= 1:
                blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
                blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
                blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(1)
    else:
        blks[0].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(1)
        blks[0].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
        blks[0].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(0)
        for k in range(nhours):
            if k >= 1:
                blks[k].usc.fs.charge_mode_disjunct.binary_indicator_var.set_value(0)
                blks[k].usc.fs.discharge_mode_disjunct.binary_indicator_var.set_value(0)
                blks[k].usc.fs.no_storage_mode_disjunct.binary_indicator_var.set_value(1)

    # Select solver and solve the model
    csvfile = create_csv_header(nhours=nhours)

    opt = pyo.SolverFactory('gdpopt')
    _prop_bnds_root_to_leaf_map[ExternalFunctionExpression] = lambda x, y, z: None

    net_power = []
    hot_tank_level = []
    cold_tank_level = []
    hxc_duty = []
    hxd_duty = []
    boiler_heat_duty = []
    discharge_work = []
    for week in range(nweeks):
        print()
        print(">> Solving for week {}: {} hours of operation in {} day(s) ".
              format(week + 1, nhours, ndays))
        results = opt.solve(
            m,
            tee=True,
            algorithm='RIC',
            mip_solver='gurobi_direct',
            nlp_solver='ipopt',
            # # OA_penalty_factor=1e4,
            # # max_slack=1e4,
            # zero_tolerance=1e-10,
            # integer_tolerance=1e-4,
            # variable_tolerance=1e-6,
            init_algorithm="no_init",
            subproblem_presolve=False,
            time_limit="56000",
            iterlim=500,
            call_after_subproblem_solve=(
                lambda c, a, b: print_model(c, a, b,
                                            csvfile, nweeks=nweeks,
                                            nhours=nhours, lmp=lmp,
                                            n_time_points=n_time_points)
            ),
            nlp_solver_args=dict(
                tee=True,
                symbolic_solver_labels=True,
                options={"linear_solver": "ma27",
                         "max_iter": 150,
                         "halt_on_ampl_error": "yes"
                }
            )
        )

        hot_tank_level.append([pyo.value(blks[i].usc.salt_inventory_hot) # in mton
                               for i in range(n_time_points)])
        cold_tank_level.append([pyo.value(blks[i].usc.salt_inventory_cold) # in mton
                                for i in range(n_time_points)])
        net_power.append([pyo.value(blks[i].usc.fs.net_power)
                          for i in range(n_time_points)])
        boiler_heat_duty.append([pyo.value(blks[i].usc.fs.boiler.heat_duty[0]) * 1e-6
                                 for i in range(n_time_points)])
        discharge_work.append([pyo.value(blks[i].usc.fs.discharge_turbine_work)
                               for i in range(n_time_points)])
        hxc_duty.append([pyo.value(blks[i].usc.fs.charge_mode_disjunct.indicator_var) *
                         pyo.value(blks[i].usc.fs.charge_mode_disjunct.hxc.heat_duty[0]) * 1e-6
                         for i in range(n_time_points)])
        hxd_duty.append([pyo.value(blks[i].usc.fs.discharge_mode_disjunct.indicator_var) *
                         pyo.value(blks[i].usc.fs.discharge_mode_disjunct.hxd.heat_duty[0]) * 1e-6
                         for i in range(n_time_points)])

    csvfile.close()

    return (m,
            blks,
            lmp,
            net_power,
            results,
            hot_tank_level,
            cold_tank_level,
            hxc_duty,
            hxd_duty,
            boiler_heat_duty,
            discharge_work)

def print_results(m, blks, results):
    # Print and plot results
    c = 0
    print('Objective: {:.4f}'.format(value(m.obj) / scaling_obj))
    for blk in blks:
        print()
        print('Period {}'.format(c+1))
        storage_work = blks[c].usc.fs.discharge_turbine_work
        charge_mode = blks[c].usc.fs.charge_mode_disjunct
        discharge_mode = blks[c].usc.fs.discharge_mode_disjunct
        perc = 100
        factor = 1

        print(' Charge mode: {}'.format(
            blks[c].usc.fs.charge_mode_disjunct.binary_indicator_var.value))
        print(' Discharge mode: {}'.format(
            blks[c].usc.fs.discharge_mode_disjunct.binary_indicator_var.value))
        print(' No storage mode: {}'.format(
            blks[c].usc.fs.no_storage_mode_disjunct.binary_indicator_var.value))
        if blks[c].usc.fs.charge_mode_disjunct.binary_indicator_var.value == 1:
            print('  HXC area (m2): {:.4f}'.format(
                value(charge_mode.hxc.area)))
            print('  HXC Duty (MW): {:.4f}'.format(
                value(charge_mode.hxc.heat_duty[0]) * 1e-6))
            print('  HXC salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(charge_mode.hxc.tube_inlet.temperature[0]),
                value(charge_mode.hxc.tube_outlet.temperature[0])))
            print('  HXC steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(charge_mode.hxc.hot_side.properties_in[0].temperature),
                value(charge_mode.hxc.hot_side.properties_out[0].temperature)))
            print('  HXC salt flow (kg/s) [mton/h]: {:.4f} [{:.4f}]'.format(
                value(charge_mode.hxc.tube_outlet.flow_mass[0]),
                value(charge_mode.hxc.tube_outlet.flow_mass[0]) * 3600 * factor_mton))
            print('  HXC steam flow (mol/s): {:.4f}'.format(
                value(charge_mode.hxc.shell_outlet.flow_mol[0])))
            print('  HXC Delta T (K): in: {:.4f}, out: {:.4f}'.format(
                value(charge_mode.hxc.delta_temperature_in[0]),
                value(charge_mode.hxc.delta_temperature_out[0])))
        elif blks[c].usc.fs.discharge_mode_disjunct.binary_indicator_var.value == 1:
            print('  HXD area (m2): {:.4f}'.format(
                value(discharge_mode.hxd.area)))
            print('  HXD Duty (MW): {:.4f}'.format(
                value(discharge_mode.hxd.heat_duty[0]) * 1e-6))
            print('  HXD salt temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(discharge_mode.hxd.shell_inlet.temperature[0]),
                value(discharge_mode.hxd.shell_outlet.temperature[0])))
            print('  HXD steam temperature (K) in/out: {:.4f}/{:.4f}'.format(
                value(discharge_mode.hxd.cold_side.properties_in[0].temperature),
                value(discharge_mode.hxd.cold_side.properties_out[0].temperature)))
            print('  HXD salt flow (kg/s) [mton/h]: {:.4f} [{:.4f}]'.format(
                value(discharge_mode.hxd.shell_outlet.flow_mass[0]),
                value(discharge_mode.hxd.shell_outlet.flow_mass[0]) * 3600 * factor_mton))
            print('  HXD steam flow (mol/s): {:.4f}'.format(
                value(discharge_mode.hxd.tube_outlet.flow_mol[0])))
            print('  HXD Delta T (K): in: {:.4f}, out: {:.4f}'.format(
                value(discharge_mode.hxd.delta_temperature_in[0]),
                value(discharge_mode.hxd.delta_temperature_out[0])))
            print('  ES turbine work (MW): {:.4f}'.format(
                value(discharge_mode.es_turbine.work_mechanical[0]) * -1e-6))
        elif blks[c].usc.fs.no_storage_mode_disjunct.binary_indicator_var.value == 1:
            print('  **Note: no storage heat exchangers exist, so the units have the init values ')
            print('  HXC area (m2): {:.4f}'.format(
                value(charge_mode.hxc.area)))
            print('  HXC Duty (MW): {:.4f}'.format(
                value(charge_mode.hxc.heat_duty[0]) * 1e-6))
            print('  HXC salt flow (kg/s): {:.4f} '.format(
                value(charge_mode.hxc.tube_outlet.flow_mass[0])))
            print('  HXD area (m2): {:.4f}'.format(
                value(discharge_mode.hxd.area)))
            print('  HXD Duty (MW): {:.4f}'.format(
                value(discharge_mode.hxd.heat_duty[0]) * 1e-6))
            print('  HXD salt flow (kg/s): {:.4f}'.format(
                value(discharge_mode.hxd.shell_outlet.flow_mass[0])))
        else:
            print('  No other operation modes!')

        print(' Net power: {:.4f}'.format(
            value(blks[c].usc.fs.net_power)))
        print(' Plant Power Out: {:.4f}'.format(
            value(blks[c].usc.fs.plant_power_out[0])))
        print(' Discharge turbine work (MW): {:.4f}'.format(
            value(storage_work) * factor))
        # print(' Profit ($): {:.4f}'.format(
        #     value(blks[c].profit) / scaling_cost))
        # print(' Revenue ($): {:.4f}'.format(
        #     value(blks[c].revenue)))
        # print(' Operating cost ($): {:.4f}'.format(
        #     value(blks[c].total_cost)))
        print(' Efficiencies (%): boiler: {:.4f}, cycle: {:.4f}'.format(
            value(blks[c].usc.fs.boiler_efficiency) * 100,
            value(blks[c].usc.fs.cycle_efficiency) * perc))
        print(' Boiler heat duty: {:.4f}'.format(
            value(blks[c].usc.fs.boiler.heat_duty[0]) * 1e-6))
        print(' Boiler flow mol (mol/s): {:.4f}'.format(
            value(blks[c].usc.fs.boiler.outlet.flow_mol[0])))
        print(' Hot salt inventory (mton): previous: {:.4f}, current: {:.4f}'.format(
            value(blks[c].usc.previous_salt_inventory_hot),
            value(blks[c].usc.salt_inventory_hot)))
        print(' Cold salt inventory (mton): previous: {:.4f}, current: {:.4f}'.format(
            value(blks[c].usc.previous_salt_inventory_cold),
            value(blks[c].usc.salt_inventory_cold)))
        c += 1

    print(results)

def plot_results(m,
                 blks,
                 lmp,
                 ndays=None,
                 nweeks=None,
                 n_time_points=None,
                 net_power=None,
                 tank_max=None,
                 hot_tank_level=None,
                 cold_tank_level=None,
                 hxc_duty=None,
                 hxd_duty=None,
                 boiler_heat_duty=None,
                 discharge_work=None):


    c = ['darkred', 'midnightblue', 'tab:green', 'k', 'gray']
    hours = np.arange(n_time_points * nweeks)
    lmp_array = np.asarray(lmp[0:n_time_points])
    hot_tank_array = np.asarray(hot_tank_level[0:nweeks]).flatten()
    cold_tank_array = np.asarray(cold_tank_level[0:nweeks]).flatten()

    # First, convert array to list to include the
    # value at period zero, which for this analysis is zero since the
    # plant is not operating.
    hot_tank_array0 = value(blks[0].usc.previous_salt_inventory_hot)
    cold_tank_array0 = value(blks[0].usc.previous_salt_inventory_cold)
    hours_list = hours.tolist() + [nhours]
    hot_tank_list = [hot_tank_array0] + hot_tank_array.tolist()
    cold_tank_list = [cold_tank_array0] + cold_tank_array.tolist()
    hxc_array = np.asarray(hxc_duty[0:nweeks]).flatten()
    hxd_array = np.asarray(hxd_duty[0:nweeks]).flatten()
    hxc_duty_list = [0] + hxc_array.tolist()
    hxd_duty_list = [0] + hxd_array.tolist()
    boiler_heat_duty_array = np.asarray(boiler_heat_duty[0:nweeks]).flatten()
    boiler_heat_duty_list = [0] + boiler_heat_duty_array.tolist()
    power_array = np.asarray(net_power[0:nweeks]).flatten()
    power_array0 = value(blks[0].usc.previous_power)
    power_list = [power_array0] + power_array.tolist()
    discharge_work_array = np.asarray(discharge_work[0:nweeks]).flatten()
    discharge_work_list = [0] + discharge_work_array.tolist()

    # Plot molten salt tank levels for each period. First, convert
    # array to list to include hot tank level at initial period zero.
    fig3, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Amount (metric ton)', color=c[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    plt.axhline(tank_max, ls=':', lw=1.5, color=c[4])
    ax1.step(hours_list, hot_tank_list, marker='o', ms=8, lw=1.5, color=c[0], alpha=0.85,
             label='Hot Tank')
    ax1.fill_between(hours_list, hot_tank_list, step="pre", color=c[0], alpha=0.35)
    ax1.step(hours_list, cold_tank_list, marker='o', ms=8, lw=1.5, color=c[1], alpha=0.65,
             label='Cold Tank')
    ax1.fill_between(hours_list, cold_tank_list, step="pre", color=c[1], alpha=0.10)
    ax1.legend(loc="upper left", frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=1))
    ax2 = ax1.twinx()
    ax2.set_ylim((-25, 225))
    ax2.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax2.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax2.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/final_salt_tank_level.png'.format(
        nhours))

    # Plot charge and discharge heat exchangers heat duty values for
    # each time period.
    fig4, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Heat Duty (MW)', color=c[3])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    ax3.set_ylim((-25, 825))
    ax3.step(hours_list, boiler_heat_duty_list, marker='o', ms=8, ls='-', lw=1.5, alpha=0.85, color=c[3],
             label='Boiler')
    ax3.fill_between(hours_list, boiler_heat_duty_list, step="pre", color=c[3], alpha=0.15)
    plt.axhline(max_storage_heat_duty, ls=':', lw=1.5, color=c[4])
    # plt.axhline(min_storage_heat_duty, ls=':', lw=1.5, color=c[4])
    ax3.step(hours_list, hxc_duty_list, marker='o', ms=8, color=c[0], alpha=0.75,
             label='Charge')
    ax3.fill_between(hours_list, hxc_duty_list, step="pre", color=c[0], alpha=0.25)
    ax3.step(hours_list, hxd_duty_list, marker='o', ms=8, color=c[1], alpha=0.75,
             label='Discharge')
    ax3.fill_between(hours_list, hxd_duty_list, step="pre", color=c[1], alpha=0.25)
    ax3.legend(loc="center left", frameon=False)
    ax3.tick_params(axis='y')
    ax3.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=1))
    ax4 = ax3.twinx()
    ax4.set_ylim((-25, 225))
    ax4.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax4.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax4.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/final_heat_duty.png'.format(nhours))

    # Plot net power and discharge power production for each period.
    fig4, ax5 = plt.subplots(figsize=(12, 8))
    ax5.set_xlabel('Time Period (hr)')
    ax5.set_ylabel('Power Output (MW)', color=c[1])
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    plt.axhline(max_power, ls=':', lw=1.5, color=c[4])
    ax5.step(hours_list, power_list, marker='o', ms=8, lw=1.5, color=c[3],
             label='Plant Net Power')
    ax5.fill_between(hours_list, power_list, step="pre", color=c[3], alpha=0.15)
    ax5.step(hours_list, discharge_work_list, marker='o', ms=8, color=c[1], alpha=0.85,
             label='Discharge Turbine')
    ax5.fill_between(hours_list, discharge_work_list, step="pre", color=c[1], alpha=0.15)
    ax5.tick_params(axis='y', labelcolor=c[1])
    ax5.set_xticks(np.arange(0, n_time_points*nweeks + 1, step=2))
    ax6 = ax5.twinx()
    ax2.set_ylim((-25, 225))
    ax6.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax6.step([x + 1 for x in hours], lmp_array, marker='o', ms=8, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax6.tick_params(axis='y', labelcolor=c[2])
    plt.savefig('results/gdp_mp_unfixed_area_{}h/final_power.png'.format(nhours))

    plt.show()

def _mkdir(dir):
    """Create directory to save results

    """

    try:
        os.mkdir(dir)
        print('Directory {} created'.format(dir))
    except:
        print('Directory {} not created because it already exists!'.format(dir))
        pass


if __name__ == '__main__':

    optarg = {
        "max_iter": 300,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    # Save results in a .csv file for each master iteration
    save_results = True

    # Use GDP design for charge and discharge heat exchanger from 4-12
    # disjunctions model when True. If False, use the GDP design from
    # 4-5 disjunctions model. **Note** When changing this, make sure
    # to change it in the GDP multiperiod python script too.
    new_design = True

    lx = True
    if lx:
        if new_design:
            # scaling_obj = 1e-2
            # # scaling_cost = 1e-3 # before changing the obj function
            # scaling_cost = 1e-3
            scaling_obj = 1e-5
            # scaling_cost = 1e-3 # before changing the obj function
            scaling_cost = 1
        else:
            # scaling_obj = 1e-2 # 6h, 12h
            scaling_obj = 1e-4 # 12h
            scaling_cost = 1e-3
    else:
        scaling_obj = 1
        scaling_cost = 1
    print()
    print('Scaling cost:', scaling_cost)
    print('Scaling obj:', scaling_obj)

    # Add design data from .json file
    if new_design:
        data_path = 'uscp_design_data_new_storage_design.json'
    else:
        data_path = 'uscp_design_data.json'

    with open(data_path) as design_data:
        design_data_dict = json.load(design_data)

    max_salt_amount = design_data_dict["max_salt_amount"] # in kg
    max_storage_heat_duty = design_data_dict["max_storage_heat_duty"] # in MW
    min_storage_heat_duty = design_data_dict["min_storage_heat_duty"] # in MW
    factor_mton = design_data_dict["factor_mton"] # factor for conversion kg to metric ton
    max_power = design_data_dict["plant_max_power"] # in MW
    pmin = design_data_dict["plant_min_power"] # in MW

    hours_per_day = 12
    ndays = 1
    nhours = hours_per_day * ndays
    nweeks = 1

    # Add number of hours per week
    n_time_points = nweeks * nhours

    tank_status = "hot_empty"
    tank_min = 1 * factor_mton # in mton
    tank_max = max_salt_amount * factor_mton # in mton

    # Create a directory to save the results for each NLP sbproblem
    # and plots
    _mkdir('results')
    _mkdir('results/gdp_mp_unfixed_area_{}h'.format(nhours))

    (m,
     blks,
     lmp,
     net_power,
     results,
     hot_tank_level,
     cold_tank_level,
     hxc_duty,
     hxd_duty,
     boiler_heat_duty,
     discharge_work) = run_pricetaker_analysis(hours_per_day=hours_per_day,
                                               nhours=nhours,
                                               ndays=ndays,
                                               nweeks=nweeks,
                                               n_time_points=n_time_points,
                                               pmin=pmin,
                                               tank_status=tank_status,
                                               tank_min=tank_min,
                                               tank_max=tank_max)

    print_results(m,
                  blks,
                  results)

    plot_results(m,
                 blks,
                 lmp,
                 ndays=ndays,
                 nweeks=nweeks,
                 n_time_points=n_time_points,
                 hot_tank_level=hot_tank_level,
                 cold_tank_level=cold_tank_level,
                 net_power=net_power,
                 hxc_duty=hxc_duty,
                 hxd_duty=hxd_duty,
                 tank_max=tank_max,
                 boiler_heat_duty=boiler_heat_duty,
                 discharge_work=discharge_work)
