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

"""This script uses the multiperiod model for the simulatenous design
and operation of an integrated ultra-supercritical power plant with
energy storage and performs market analysis using the pricetaker
assumption. The electricity prices, LMP (locational marginal prices),
are assumed constant. The prices used in this study are either
obtained from a synthetic database or from NREL data.

"""

__author__ = "Soraya Rawlings and Naresh Susarla"

import logging

# Import Python libraries
import numpy as np
import json
import os
import pandas as pd

# Import Pyomo objects
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.environ import (Objective, Expression, value, maximize, RangeSet, Constraint)
from pyomo.repn.plugins.nl_writer import _activate_nl_writer_version
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.util.infeasible import (log_infeasible_constraints,
                                   log_close_to_bounds)
import idaes.core.util.scaling as iscale
# Import multiperiod model
from fixed_nlp_multiperiod import create_nlp_multiperiod_usc_model

# Import IDAES libraries
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers.get_solver import get_solver

from dispatches.properties import solarsalt_properties

from idaes.core import UnitModelCostingBlock
from idaes.models.costing.SSLW import (SSLWCosting, SSLWCostingData,
                                       PumpType, PumpMaterial, PumpMotorType)
from idaes.core.util.model_diagnostics import DegeneracyHunter
from idaes.core.util.scaling import (list_unscaled_constraints,
                                     list_unscaled_variables,
                                     extreme_jacobian_rows,
                                     extreme_jacobian_columns)
from idaes.core.util.model_statistics import large_residuals_set

logging.getLogger('pyomo.repn.plugins.nl_writer').setLevel(logging.ERROR)

# Import objects for plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
font = {'size':16}
plt.rc('axes', titlesize=24)
plt.rc('font', **font)

# _activate_nl_writer_version(2)

# Make sure these have the same value in nlp_multiperiod script
use_surrogate = False
constant_salt = True
fix_design = True

def _get_lmp(n_time_points=None):

    # Select lmp source data and scaling factor according to that
    use_rts_data = False
    use_mod_rts_data = True
    if use_rts_data:
        print('>>>>>> Using RTS LMP data')
        with open('rts_results_all_prices_base_case.npy', 'rb') as f:
            dispatch = np.load(f)
            price = np.load(f)
        lmp = price[0:nhours].tolist()
    elif use_mod_rts_data:
        print('>>>>>> Using (modified or avrg) RTS LMP data')
        # RTS modified data
        # price = [52.9684, 21.1168, 10.4, 5.419,
        #          20.419, 21.2877, 23.07, 25,
        #          18.4634, 0, 0, 0,
        #          0, 0, 0, 0,
        #          19.0342, 23.07, 200, 200,
        #          200, 200, 200, 200]
        if n_time_points == 24:
            # RTS average 24 hrs
            price = [
                21.734392123626375, 20.991034120879117, 19.896812835164834, 18.83252368406595,
                18.797422843406594, 20.550819736263733, 21.571611835164816, 18.6866115879121,
                11.13006721978022, 9.296121148351645, 9.218053085164833, 10.285348115384613,
                11.446111348901104, 13.139503247252758, 14.844101744505506, 17.63673195879121,
                20.526543373626396, 26.187743260989023, 33.64193449450563, 34.581440082417686,
                33.32562242857146, 29.430152887362656, 26.159412942307675, 23.720650788461544
            ]
        elif n_time_points == 3:
            # RTS average 24 hrs
            price = [
                21.734392123626375, 20.991034120879117, 19.896812835164834
            ]
        elif n_time_points == 168:
            # RTS average for 1 week
            price = [
                20.829564538461536, 19.693028192307693, 17.820727653846156, 16.635251211538467,
                16.858723211538457, 19.40059307692308, 20.643515596153847, 17.96632492307692,
                11.641621326923078, 8.882227480769231, 8.57824726923077, 10.088016769230771,
                11.98047917307692, 13.386592423076921, 15.169638807692307, 17.26273592307692,
                20.160692365384612, 26.26299409615385, 35.899514692307676, 34.625430192307675,
                33.08377317307692, 29.34617059615384, 25.27478221153846, 23.242250461538465,
                20.579725442307698, 20.10433521153846, 19.583595423076922, 18.30950607692308,
                17.745158134615387, 20.727536230769225, 21.299347000000004, 22.56708517307692,
                13.05982015384615, 11.377053153846152, 12.062396769230768, 13.014763615384611,
                13.309641346153843, 14.10522267307692, 15.761158846153844, 17.752077307692307,
                22.15561392307692, 27.434314403846148, 32.038436692307684, 34.166914615384584,
                32.15269176923077, 30.22890519230769, 26.737662596153843, 24.446553615384616,
                23.87905498076924, 23.481888096153853, 22.33163884615385, 22.013695730769232,
                22.056609076923085, 23.214242769230776, 20.77664930769231, 16.460505769230767,
                10.694727730769232, 8.226054153846153, 8.556423807692308, 9.55585592307692,
                9.452980692307694, 11.024066711538463, 12.853477865384612, 15.963486576923076,
                20.566712480769226, 25.440112192307687, 33.19064453846151, 32.40535288461537,
                32.056517980769215, 29.629065499999985, 26.52354048076922, 24.592951846153845,
                21.957720769230775, 21.656073307692303, 20.644124711538463, 19.998112538461537,
                18.79414169230769, 18.824993884615388, 18.83248876923077, 13.422123596153842,
                5.286316461538462, 4.163297365384615, 5.335777173076924, 6.24375303846154,
                7.785439769230767, 9.262351384615384, 11.446491019230766, 15.400954711538466,
                18.16094686538462, 24.058750769230773, 33.21687103846154, 32.37216078846153,
                31.566947326923067, 29.482025980769222, 26.620510826923073, 22.849809673076933,
                20.383665711538455, 19.11278348076923, 18.931816269230772, 17.965620923076923,
                18.811001730769235, 20.446076096153845, 22.172929730769226, 19.089245519230772,
                12.140012634615385, 10.328201230769228, 10.470433615384612, 11.15892865384615,
                12.674435192307689, 14.398074096153845, 15.69510803846154, 18.96061357692308,
                21.639628192307693, 26.773128942307686, 32.45396540384614, 41.09640323076923,
                32.790869153846145, 29.437534576923067, 26.267306057692316, 24.10291501923077,
                22.63920815384616, 21.559509846153848, 19.997214846153845, 18.024930365384613,
                17.87084488461539, 19.87102576923077, 23.644809192307687, 21.342310192307686,
                13.331565326923073, 12.469750769230764, 10.064728865384613, 10.204487153846152,
                11.857142999999997, 14.898623096153843, 16.153417134615385, 18.845525038461535,
                20.57578742307692, 26.75846975, 36.999615134615375, 33.925467788461525,
                38.73873730769229, 29.278822115384603, 26.149744538461544, 24.07332378846155,
                21.87180526923077, 21.32962071153846, 19.968572096153853, 18.880548942307694,
                19.445481173076924, 21.371270326923078, 23.631543249999996, 19.958685942307685,
                11.75640690384615, 9.626263884615385, 9.458364096153847, 11.731631653846154,
                13.062660269230769, 14.901592346153844, 16.829420499999994, 19.27173057692308,
                20.426422365384614, 26.586432673076917, 31.694493961538452, 33.478351076923055,
                32.889820288461515, 28.608546249999986, 25.542343884615384, 22.736751115384624
            ]
        else:
            print('RTS average data not given for {} hours'.format(n_time_points))
        lmp = price
        print('lmp:', lmp)
        if len(price) < n_time_points:
            print()
            print('**ERROR: I need more LMP data!')
            raise Exception
    else:
        print('>>>>>> Using NREL LMP data')
        price = np.load("nrel_scenario_average_hourly.npy")

    return lmp

def add_storage_hx_capital_cost(m):
    # Add IDAES costing method
    m.costing = SSLWCosting()

    # Calculate charge and discharge heat exchangers costs, estimated
    # using the IDAES costing method with default options, i.e., a
    # U-tube heat exchanger, stainless steel material, and a tube
    # length of 12ft. Refer to costing documentation to change any of
    # the default options. The purchase cost of heat exchanger has to
    # be annualized when used
    for storage_hx in [m.period[1].fs.hxc,
                       m.period[1].fs.hxd]:
        storage_hx.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.costing,
            costing_method=SSLWCostingData.cost_heat_exchanger
        )
    # m.storage_hx_capital_cost = pyo.Param(initialize=6.7978,
    #                                     # bounds=(0, 1e2),
    #                                     doc="Storage heat exchangers capital cost in $/h")
    m.storage_hx_capital_cost = pyo.Var(initialize=6.7978,
                                        # bounds=(0, 1e2),
                                        doc="Storage heat exchangers capital cost in $/h")
    def rule_storage_hx_capital_cost(b):
        return (b.storage_hx_capital_cost == (
            (b.period[1].fs.hxc.costing.capital_cost +
             b.period[1].fs.hxd.costing.capital_cost)/(b.period[1].fs.num_of_years*365*24) # heat exchangers cost
        ))
    m.eq_storage_hx_capital_cost = pyo.Constraint(rule=rule_storage_hx_capital_cost)

    calculate_variable_from_constraint(m.storage_hx_capital_cost,
                                       m.eq_storage_hx_capital_cost)
    iscale.constraint_scaling_transform(m.eq_storage_hx_capital_cost, 1e-4)
    # m.eq_storage_hx_capital_cost.pprint()
    # print("cost before initializing", value(m.storage_hx_capital_cost))
    # assert False
    for unit_i in [m.period[1].fs.hxc, m.period[1].fs.hxd]:
        calculate_variable_from_constraint(unit_i.costing.base_cost_per_unit,
                                           unit_i.costing.base_cost_per_unit_eq)
        calculate_variable_from_constraint(unit_i.costing.material_factor,
                                           unit_i.costing.hx_material_eqn)
        calculate_variable_from_constraint(unit_i.costing.pressure_factor,
                                           unit_i.costing.p_factor_eq)
        calculate_variable_from_constraint(unit_i.costing.capital_cost,
                                           unit_i.costing.capital_cost_constraint)
        iscale.constraint_scaling_transform(unit_i.costing.base_cost_per_unit_eq, 1e0)
        iscale.constraint_scaling_transform(unit_i.costing.hx_material_eqn, 1e1)
        iscale.constraint_scaling_transform(unit_i.costing.p_factor_eq, 1e1)
        iscale.constraint_scaling_transform(unit_i.costing.capital_cost_constraint, 1e-4)

    calculate_variable_from_constraint(m.storage_hx_capital_cost,
                                       m.eq_storage_hx_capital_cost)
    iscale.constraint_scaling_transform(m.eq_storage_hx_capital_cost, 1e-4)
    m.storage_hx_capital_cost.fix(value(m.storage_hx_capital_cost))
    m.period[1].fs.hxc.costing.deactivate()
    m.period[1].fs.hxd.costing.deactivate()
    m.eq_storage_hx_capital_cost.deactivate()
    # m.eq_storage_hx_capital_cost.pprint()
    # print("cost after initializing", value(m.storage_hx_capital_cost))
    # assert False


def add_storage_salt_tank_cost(m):
    """Add equations to calculate storage material tank cost considering a
    vertical vessel. For this, first calculate size and dimensions of
    Solar salt storage tank followed by the calculation of the Solar
    salt tank volume with a 10% margin. To compute the Solar salt tank
    surface area, consider the surface area of sides and top surface
    area. The base area is accounted in foundation costs

    """

    # Add data as Param for the storage tank calculation
    m.data_tank = {'LbyD': 0.325,
                   'tank_thickness': 0.039,
                   'material_density': 7800,
                   'cost_tank_material': 3.5,
                   'cost_tank_insulation': 235,
                   'cost_tank_foundation': 1210}
    m.l_by_d = pyo.Param(initialize=m.data_tank['LbyD'],
                         doc='L by D assumption for computing storage tank dimensions')
    m.tank_thickness = pyo.Param(initialize=m.data_tank['tank_thickness'],
                                 doc='Storage tank thickness assumed based on reference')
    m.tank_material_density = pyo.Param(initialize=m.data_tank['material_density'],
                                        doc='Tank material density in kg/m3')
    m.data_tank_material_cost = pyo.Param(initialize=m.data_tank['cost_tank_material'],
                                          doc='Tank SS316 material cost in $/kg')
    m.data_tank_insulation_cost = pyo.Param(initialize=m.data_tank['cost_tank_insulation'])
    m.data_tank_foundation_cost = pyo.Param(initialize=m.data_tank['cost_tank_foundation'])

    m.no_of_tanks = pyo.Param(initialize=2,
                              doc='Number of tank for storage material')
    m.tank_volume = pyo.Var(initialize=3500,
                            # bounds=(1, 5000),
                            units=pyunits.m**3,
                            doc="Volume of the Salt Tank with 10% excess capacity")
    m.tank_diameter = pyo.Var(initialize=1.0,
                            #   bounds=(0.5, 40),
                              units=pyunits.m,
                              doc="Diameter of the salt tank")
    m.tank_height = pyo.Var(initialize=1.0,
                            bounds=(0.5, 13),
                            units=pyunits.m,
                            doc="Length of the salt tank in m")
    m.tank_surf_area = pyo.Var(initialize=1000,
                            #    bounds=(1, 5000),
                               units=pyunits.m**2,
                               doc="Surface area of salt tank")
    m.tank_material_cost = pyo.Var(initialize=1e3,
                                #    bounds=(0, 1e4)
                                   )
    m.tank_insulation_cost = pyo.Var(initialize=1e3,
                                    #  bounds=(0, 1e4)
                                     )
    m.tank_foundation_cost = pyo.Var(initialize=1e3,
                                    #  bounds=(0, 1e4)
                                     )
                                     

    def rule_salt_tank_volume(b):
        return b.tank_volume == (
            m.period[1].fs.salt_amount*1e3*1.10/
            m.period[1].fs.hxc.cold_side.properties_out[0].dens_mass["Liq"] # at highest temperature
        )
    m.eq_tank_volume = pyo.Constraint(rule=rule_salt_tank_volume,
                                      doc="Volume of Solar salt tank")
    calculate_variable_from_constraint(m.tank_volume,
                                       m.eq_tank_volume)
    def rule_salt_tank_diameter(b):
        return m.tank_diameter == (4*(b.tank_volume/m.no_of_tanks)/(b.l_by_d*np.pi))**(1/3)
    m.eq_tank_diameter = pyo.Constraint(rule=rule_salt_tank_diameter,
                                        doc="Diameter of Solar salt tank for assumed lenght")
    calculate_variable_from_constraint(m.tank_diameter,
                                       m.eq_tank_diameter)
    def rule_salt_tank_height(b):
        return m.tank_height == b.l_by_d*b.tank_diameter
    m.eq_tank_height = pyo.Constraint(rule=rule_salt_tank_height,
                                      doc="Height of Solar salt tank")
    calculate_variable_from_constraint(m.tank_height,
                                       m.eq_tank_height)
    def rule_salt_tank_surf_area(b):
        return m.tank_surf_area == ((np.pi*b.tank_diameter*m.tank_height) +
                                    (np.pi*b.tank_diameter**2) / 4)
    m.eq_tank_surf_area = pyo.Constraint(rule=rule_salt_tank_surf_area,
                                         doc="Surface area of Solar salt tank")
    calculate_variable_from_constraint(m.tank_surf_area,
                                       m.eq_tank_surf_area)
    def rule_tank_material_cost(b):
        return (m.tank_material_cost*(b.period[1].fs.num_of_years*365*24) == (
            (b.data_tank_material_cost*b.tank_material_density*
             b.tank_surf_area*b.tank_thickness))
        )
    m.eq_tank_material_cost = pyo.Constraint(rule=rule_tank_material_cost,
                                             doc="Cost of tank material")
    calculate_variable_from_constraint(m.tank_material_cost,
                                       m.eq_tank_material_cost)
    def rule_tank_foundation_cost(b):
        return (m.tank_foundation_cost*(b.period[1].fs.num_of_years*365*24) == (
            (b.data_tank_foundation_cost*
             np.pi*b.tank_diameter**2/4))
        )
    m.eq_tank_foundation_cost = pyo.Constraint(rule=rule_tank_foundation_cost)

    calculate_variable_from_constraint(m.tank_foundation_cost,
                                       m.eq_tank_foundation_cost)
    def rule_tank_insulation_cost(b):
        return (m.tank_insulation_cost*(b.period[1].fs.num_of_years*365*24) == (
            b.data_tank_insulation_cost*m.tank_surf_area)
            )
    m.eq_tank_insulation_cost = pyo.Constraint(rule=rule_tank_insulation_cost)
    calculate_variable_from_constraint(m.tank_insulation_cost,
                                       m.eq_tank_insulation_cost)


    def rule_salt_tank_capital_cost(b):
        return (
            b.tank_material_cost +
            b.tank_foundation_cost +
            b.tank_insulation_cost
        )
    m.salt_tank_capital_cost = pyo.Expression(rule=rule_salt_tank_capital_cost,
                                              doc="Capital cost for Solar salt tank")

    iscale.constraint_scaling_transform(m.eq_tank_volume, 1e-1)
    iscale.constraint_scaling_transform(m.eq_tank_diameter, 1e1)
    iscale.constraint_scaling_transform(m.eq_tank_height, 1e0)
    iscale.constraint_scaling_transform(m.eq_tank_surf_area, 1e0)
    iscale.constraint_scaling_transform(m.eq_tank_material_cost, 1e-4)
    iscale.constraint_scaling_transform(m.eq_tank_foundation_cost, 1e-4)
    iscale.constraint_scaling_transform(m.eq_tank_insulation_cost, 1e-4)


def run_pricetaker_analysis(nweeks=None,
                            n_time_points=None,
                            pmin=None,
                            pmax=None,
                            tank_status=None,
                            max_salt_amount=None,
                            constant_salt=None):

    # Get LMP data
    lmp = _get_lmp(n_time_points=n_time_points)

    # Create the multiperiod model object. You can pass arguments to your
    # "process_model_func" for each time period using a dict of dicts as
    # shown here.  In this case, it is setting up empty dictionaries for
    # each time period.
    m = create_nlp_multiperiod_usc_model(
        n_time_points=n_time_points,
        pmin=pmin,
        pmax=pmax
    )

    # Retrieve pyomo model and active process blocks (i.e. time blocks)
    blks = [m.period[t] for t in m.set_period]

    @m.Constraint()
    def rule_periodic_variable_pair(b):
        return (b.period[1].fs.previous_salt_inventory_hot ==
                b.period[n_time_points].fs.salt_inventory_hot)
        # return (b.period[n_time_points].fs.salt_inventory_hot ==
        #         b.period[1].fs.previous_salt_inventory_hot)

    iscale.constraint_scaling_transform(m.rule_periodic_variable_pair, 1e-1)
    calculate_variable_from_constraint(m.period[n_time_points].fs.salt_inventory_hot,
                                       m.rule_periodic_variable_pair)

    ##################################################################
    # Add nonanticipativity constraints
    ##################################################################
    m.hours_set = RangeSet(1, n_time_points)
    m.hours_set2 = RangeSet(1, n_time_points - 1)

    for h in m.hours_set2:
        iscale.constraint_scaling_transform(m.link_constraints[h].link_constraints[0], 1e-1)
        iscale.constraint_scaling_transform(m.link_constraints[h].link_constraints[1], 1e-1)
    # if not fix_design:
    #     # Add nonanticipativaty constraints for charge and
    #     # discharge areas
    #     @m.Constraint(m.hours_set2)
    #     def nonanticipativity_constraint_charge_area(b, h):
    #         return b.period[h+1].fs.hxc.area == b.period[h].fs.hxc.area

    #     @m.Constraint(m.hours_set2)
    #     def nonanticipativity_constraint_discharge_area(b, h):
    #         return b.period[h+1].fs.hxd.area == b.period[h].fs.hxd.area

    #     # Add nonanticipative constraints to ensure that the discharge
    #     # heat exchanger has the same temperature for the hot salt than
    #     # the one obtained during charge cycle.
    #     @m.Constraint(m.hours_set2,
    #                   doc="Salt temperature in charge heat exchanger")
    #     def constraint_charge_temperature(b, h):
    #         return b.period[h+1].fs.hxc.tube_outlet.temperature[0] == (
    #             b.period[h].fs.hxc.tube_outlet.temperature[0]
    #         )
    #     @m.Constraint(m.hours_set,
    #                   doc="Salt temperature in charge heat exchanger")
    #     def constraint_discharge_temperature(b, h):
    #         return b.period[h].fs.hxd.shell_inlet.temperature[0] == (
    #             b.period[h].fs.hxc.tube_outlet.temperature[0]
    #         )
    # # Add constraint to ensure a hot salt temperature close to the
    # # upper bound
    # @m.Constraint(m.hours_set)
    # def rule_fix_hot_salt(b, h):
    #     return b.period[h].fs.hxc.tube_outlet.temperature[0] == 826.56*pyunits.K

    ##################################################################
    # Add storage material capital costs and inventory balances      #
    ##################################################################

    # m.max_inventory = pyo.units.convert(1e7*pyunits.kg,
    #                                     to_units=pyunits.metric_ton)
    # m.total_inventory = pyo.Var(initialize=max_salt_amount,
    #                             bounds=(0, m.max_inventory),
    #                             units=pyunits.metric_ton,
    #                             doc="Total inventory of salt")

    # if constant_salt:
    # m.total_inventory.fix(m.period[1].fs.salt_amount)
    # else:
    #     @m.Constraint(m.hours_set)
    #     def rule_salt_inventory_per_time(b, h):
    #         return b.period[h].fs.salt_amount <= m.total_inventory

    m.tank_amount = pyo.units.convert(6840520.68*pyunits.kg,
                                    to_units=pyunits.metric_ton)
    # m.total_inventory_mass = pyo.units.convert(m.total_inventory, to_units=pyunits.kg)
    m.solar_salt_price = pyo.Param(initialize=490, doc="Solar salt price in $/metric_ton")
    @m.Expression()
    def salt_purchase_cost(b):
        return (m.tank_amount*b.solar_salt_price)/(b.period[1].fs.num_of_years*365*24)

    # Add storage capital costs including salt and storage heat
    # exchangers units for both, charge and discharge operation
    add_storage_salt_tank_cost(m)
    add_storage_hx_capital_cost(m)

    ##################################################################
    # Add initial state for salt inventory                           #
    ##################################################################

    # Initial state for power and salt inventory linking
    # variables. Different tank scenarios are included for the Solar
    # salt tank levels and the previous tank level of the tank is
    # based on that.
    # m.tank_init = pyo.units.convert(10000*pyunits.kg,
    m.tank_init = pyo.units.convert(200000*pyunits.kg,
                                    to_units=pyunits.metric_ton)
    # @m.Constraint()
    # def power_init(b):
    #     return m.period[1].fs.previous_power == 447.66
    # m.period[1].fs.previous_power.fix(425.8765)
    # m.period[1].fs.previous_salt_inventory_hot.fix(m.tank_init)
    # m.period[1].fs.previous_salt_inventory_cold.fix(m.tank_amount - m.tank_init)
    m.period[1].fs.previous_power.fix()
    m.period[1].fs.previous_salt_inventory_hot.fix()
    m.period[1].fs.previous_salt_inventory_cold.fix()
    print("Previous Power", value(m.period[1].fs.previous_power))
    for blk in blks:
        iscale.constraint_scaling_transform(blk.fs.eq_turbine_temperature_in, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.plant_min_power_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.plant_max_power_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.net_plant_max_power_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.charge_storage_lb_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.charge_storage_ub_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.discharge_storage_lb_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.discharge_storage_ub_eq, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.constraint_ramp_down, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.constraint_ramp_up, 1e-1)
        iscale.constraint_scaling_transform(blk.fs.constraint_salt_inventory_hot, 1e-2)
        iscale.constraint_scaling_transform(blk.fs.constraint_salt_inventory, 1e2)
        iscale.constraint_scaling_transform(blk.fs.constraint_salt_maxflow_hot, 1e-2)
        iscale.constraint_scaling_transform(blk.fs.constraint_salt_maxflow_cold, 1e-2)

    # if tank_status == "hot_empty":
    #     # @m.Constraint()
    #     # def tank_init_amount(b):
    #     #     # return m.period[1].fs.previous_salt_inventory_hot <= m.tank_init
    #     #     return m.period[1].fs.previous_salt_inventory_hot == m.tank_init
    #     m.period[1].fs.previous_salt_inventory_hot.fix(m.tank_init)

        # @m.Constraint()
        # def rule_storage_previous_cold_salt(b):
        #     return m.period[1].fs.previous_salt_inventory_cold == (
        #         m.period[1].fs.salt_amount - b.tank_init
        #     )
        # calculate_variable_from_constraint(m.period[1].fs.previous_salt_inventory_cold,
        #                                 m.rule_storage_previous_cold_salt)

    ##################################################################
    # Add total costs and objective function                         #
    ##################################################################

    count = 0
    for blk in blks:
        # Add expression to calculate total profit in $/hour
        blk.total_profit = pyo.Expression(
            expr=(
                lmp[count]*blk.fs.net_power[0] - # revenue
                (
                    blk.fs.fuel_cost +
                    blk.fs.plant_operating_cost
                ) - # plant operating costs
                (
                    m.salt_purchase_cost +
                    m.no_of_tanks*m.salt_tank_capital_cost +
                    m.storage_hx_capital_cost
                ) # storage capital costs
            )
        )
        count += 1

    m.obj = pyo.Objective(
        expr=(sum([blk.total_profit for blk in blks]))*scaling_obj,
        sense=maximize
    )


    # print("  ")
    # print("  ")
    # print("Print out all constraints with residuals before calling solve")
    # lrs = large_residuals_set(m)
    # print("  ")
    # print("  ")
    # for i in lrs:
    #     print(i.name)

    # print("  ")
    # print("  ")
    # lst_c = list_unscaled_constraints(m)
    # for i in lst_c:
    #     print(i.name)

    # print("Print constraint variables for residuals")
    # m.link_constraints[1].link_constraints[0].pprint()
    # print("Period 1 Hot salt level", value(m.period[1].fs.salt_inventory_hot))
    # print("Period 2 Previous Hot salt level", value(m.period[2].fs.previous_salt_inventory_hot))
    # m.link_constraints[1].link_constraints[1].pprint()
    # print("Period 1 Power", value(m.period[1].fs.plant_power_out[0.0]))
    # print("Period 2 Previous Previous Power", value(m.period[2].fs.previous_power))
    # m.link_constraints[23].link_constraints[0].pprint()
    # print("Period 23 Hot salt level", value(m.period[23].fs.salt_inventory_hot))
    # print("Period 24 Previous Hot salt level", value(m.period[24].fs.previous_salt_inventory_hot))
    # m.link_constraints[23].link_constraints[1].pprint()
    # print("Period 23 Power", value(m.period[23].fs.plant_power_out[0.0]))
    # print("Period 24 Previous Previous Power", value(m.period[24].fs.previous_power))
    # m.period[24].fs.constraint_salt_inventory_hot.pprint()
    # m.period[24].fs.constraint_salt_inventory.pprint()
    # print("Solve with 0 Iterations")
    # # assert False
    # # Declare the solver and a set of lists to save the results
    opt = pyo.SolverFactory('ipopt')
    # print()
    # print(">>Solve for {} hours of operation ({} day(s)) ".format(n_time_points,
    #                                                               n_time_points/24))
    # # print("  ")
    # # print("  ")
    # # print("Solve with 0 Iterations")
    # results = opt.solve(m,
    #                     tee=True,
    #                     symbolic_solver_labels=True,
    #                     options={
    #                         "max_iter": 0,
    #                         "linear_solver": "ma57",
    #                         "bound_push": 1e-6,
    #                         # "halt_on_ampl_error": "yes"
    #                     })
    # print("  ")
    # print("  ")
    # print("Print out all constraints with residuals larger than 0.1")
    # lrs = large_residuals_set(m)
    # for i in lrs:
    #     print(i.name)
    # # assert False
    # milp_solver =  pyo.SolverFactory('gurobi')
    # dh = DegeneracyHunter(m, solver=milp_solver)

    # print("  ")
    # print("  ")
    # print("Print out all constraints with residuals larger than 0.1")
    # dh.check_residuals(tol=1e-5)

    # print("  ")
    # print("  ")
    # print("Listing Extreme jacobian Rows")
    # print("  ")
    # print("  ")
    # lst_r = extreme_jacobian_rows(m, scaled=True)
    # for i in lst_r:
    #     print(i[0], i[1].name)
    # print("  ")
    # print("  ")
    # print("Listing Extreme jacobian Columns")
    # lst_cv = extreme_jacobian_columns(m, scaled=True)
    # for i in lst_cv:
    #     print(i[0], i[1].name)
    print("  ")
    print("  ")
    # print("  ")
    # print("  ")
    # print("Which variables are within their bounds by a given tolerance?")
    # dh.check_variable_bounds(tol=1.0)

    print("  ")
    print("  ")
    # print("Solve with 50 Iterations")
    results = opt.solve(m,
                        tee=True,
                        symbolic_solver_labels=True,
                        options={
                            "max_iter": 300,
                            "linear_solver": "ma57",
                            "halt_on_ampl_error": "yes"
                        })
    # print("  ")
    # print("  ")
    # print("Check the rank of the Jacobian of the equality constraints")
    # n_deficient = dh.check_rank_equality_constraints()
    # print("  ")
    # print("  ")
    # print(" Identify candidate degenerate constraints")
    # ds = dh.find_candidate_equations(verbose=True,tee=True)
    # print("  ")
    # print("  ")
    # print(" Find irreducible degenerate sets")
    # ids = dh.find_irreducible_degenerate_sets(verbose=True)

    hot_tank_level = []
    cold_tank_level = []
    net_power = []
    hxc_duty = []
    hxd_duty = []
    plant_heat_duty = []
    discharge_work = []
    total_inventory = []
    steam_to_storage = []
    boiler_flow = []
    boiler_heat_duty = []
    fuel_cost = []
    plant_operating_cost = []
    salt_cost = []
    tank_cost = []
    hx_cost = []
    for blk in blks:
        # Save results in lists
        hot_tank_level.append(pyo.value(blk.fs.salt_inventory_hot))
        cold_tank_level.append(pyo.value(blk.fs.salt_inventory_cold))
        plant_heat_duty.append(pyo.value(blk.fs.plant_heat_duty[0])) # in MW
        discharge_work.append(pyo.value(blk.fs.es_turbine.work[0])*(-1e-6)) # in MW
        net_power.append(pyo.value(blk.fs.net_power[0]))
        hxc_duty.append(pyo.value(blk.fs.hxc.heat_duty[0])*1e-6)
        hxd_duty.append(pyo.value(blk.fs.hxd.heat_duty[0])*1e-6)
        fuel_cost.append(pyo.value(blk.fs.fuel_cost))
        plant_operating_cost.append(pyo.value(blk.fs.plant_operating_cost))
        salt_cost.append(pyo.value(m.salt_purchase_cost))
        tank_cost.append(pyo.value(m.no_of_tanks*m.salt_tank_capital_cost))
        hx_cost.append(pyo.value(m.storage_hx_capital_cost))
        total_inventory.append(pyo.value(m.period[1].fs.salt_amount))

        if use_surrogate:
            steam_to_storage.append(pyo.value(blk.fs.steam_to_storage[0]))
            boiler_flow.append(pyo.value(blk.fs.boiler_flow[0]))
        else:
            steam_to_storage.append(pyo.value(blk.fs.hxc.shell_inlet.flow_mol[0]))
            boiler_flow.append(pyo.value(blk.fs.boiler.inlet.flow_mol[0]))

    log_close_to_bounds(m)
    log_infeasible_constraints(m)
    df_results = pd.DataFrame.from_dict({
        "hot_tank_level": hot_tank_level,
        "cold_tank_level": cold_tank_level,
        "plant_heat_duty": plant_heat_duty,
        "discharge_work": discharge_work,
        "net_power": net_power,
        "hxc_duty": hxc_duty,
        "hxd_duty": hxd_duty,
        "fuel_cost": fuel_cost,
        "plant_operating_cost": plant_operating_cost,
        "salt_cost": salt_cost,
        "tank_cost": tank_cost,
        "hx_cost": hx_cost,
        "lmp": lmp
    }
    )
    df_results.to_excel("results_output_fixed_0530.xlsx")
    
    print('hot_tank_level=', hot_tank_level)
    print('cold_tank_level=', cold_tank_level)
    print('boiler_flow=', boiler_flow)
    print('steam_to_storage=', steam_to_storage)
    print('plant_heat_duty=', plant_heat_duty)
   
    if not use_surrogate:
        for blk in blks:
            boiler_heat_duty.append(pyo.value(blk.fs.boiler.heat_duty[0])*1e-6) # in MW
    
        print('boiler_heat_duty=', boiler_heat_duty)

    # return m
    return (m, blks, lmp, net_power, results,
            total_inventory, hot_tank_level,
            cold_tank_level, hxc_duty, hxd_duty,
            boiler_heat_duty,
            plant_heat_duty, discharge_work)


def print_results(m, blks, results):
    c = 0
    print('Objective: {:.4f}'.format(pyo.value(m.obj)/scaling_obj))
    for blk in blks:
        print()
        print('{}'.format(blk))
        print(' Net power: {:.4f}'.format(pyo.value(blk.fs.net_power[0])))
        print(' Plant power out (MW): {:.4f}'.format(pyo.value(blk.fs.plant_power_out[0])))
        print(' Plant heat duty (MW): {:.4f}'.format(pyo.value(blk.fs.plant_heat_duty[0])))
        print(' Coal heat duty (MW): {:.4f}'.format(pyo.value(blk.fs.coal_heat_duty)))
        print(' ES turbine power (MW): {:.4f}'.format(
            pyo.value(blk.fs.es_turbine.work_mechanical[0])*(-1e-6)))
        print(' HX pump power (MW): {:.4f}'.format(
            pyo.value(blk.fs.hx_pump.control_volume.work[0])*(1e-6)))
        print(' Boiler/cycle efficiency (%): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.boiler_efficiency)*100,
            pyo.value(blk.fs.cycle_efficiency)*100))
        print(' Costs')
        # print(' Revenue ($/h): {:.4f}'.format(pyo.value(blk.revenue)))
        print('  Fuel cost ($/h): {:.4f}'.format(pyo.value(blk.fs.fuel_cost)))
        # print(' Plant capital cost ($/h): {:.4f}'.format(pyo.value(blk.fs.plant_capital_cost)))
        print('  Plant operating cost ($/h): {:.4f}'.format(pyo.value(blk.fs.plant_operating_cost)))
        print('  Storage HXC capital cost ($/h): {:.4f}'.format(
            pyo.value(m.period[1].fs.hxc.costing.capital_cost)))
        print('  Storage HXD capital cost ($/h): {:.4f}'.format(
            pyo.value(m.period[1].fs.hxd.costing.capital_cost)))
        print('  Storage HX capital cost ($/h): {:.4f}'.format(
            pyo.value(m.storage_hx_capital_cost)))
        print('  Storage purchase salt capital cost ($/h): {:.4f}'.format(
            pyo.value(m.salt_purchase_cost)))
        print('  Storage tank salt capital cost ($/h): {:.4f}'.format(
            pyo.value(m.salt_tank_capital_cost)))
        print(' Salt tank')
        print('  Volume: {:.4f}'.format(pyo.value(m.tank_volume)))
        print('  Diameter: {:.4f}'.format(pyo.value(m.tank_diameter)))
        print('  Surf area: {:.4f}'.format(pyo.value(m.tank_surf_area)))
        print('  Insulation cost: {:.4f}'.format(pyo.value(m.tank_insulation_cost)))
        print('  Material cost: {:.4f}'.format(pyo.value(m.tank_material_cost)))
        print('  Foundation cost: {:.4f}'.format(pyo.value(m.tank_foundation_cost)))
        if use_surrogate:
            print(' Boiler flow mol (mol/s): {:.4f}'.format(pyo.value(blk.fs.boiler_flow[0])))
        else:
            print(' Boiler heat duty: {:.4f}'.format(pyo.value(blk.fs.boiler.heat_duty[0])*1e-6))
            print(' Boiler flow mol (mol/s): {:.4f}'.format(
                pyo.value(blk.fs.boiler.outlet.flow_mol[0])))
        print(' Salt inventory (mton)')
        print('  Total inventory: {:.4f}'.format(pyo.value(m.period[1].fs.salt_amount)))
        print('  Salt amount: {:.4f}'.format(pyo.value(blk.fs.salt_amount)))
        print('  Hot salt (previous/now): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.previous_salt_inventory_hot),
            pyo.value(blk.fs.salt_inventory_hot)))
        print('  Cold salt (previous/now): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.previous_salt_inventory_cold),
            pyo.value(blk.fs.salt_inventory_cold)))
        print(' Salt flow HXC (mton/h) [kg/s]: {:.4f} [{:.4f}]'.format(
            pyo.value(blk.fs.hxc.tube_inlet.flow_mass[0])*3600*1e-3,
            pyo.value(blk.fs.hxc.tube_outlet.flow_mass[0])))
        print(' Salt flow HXD (mton/h) [kg/s]: {:.4f} [{:.4f}]'.format(
            pyo.value(blk.fs.hxd.shell_inlet.flow_mass[0])*3600*1e-3,
            pyo.value(blk.fs.hxd.shell_outlet.flow_mass[0])))
        print(' Steam flow HXC (mol/s): {:.4f}'.format(
            pyo.value(blk.fs.hxc.shell_outlet.flow_mol[0])))
        print(' Steam flow HXD (mol/s): {:.4f}'.format(
            pyo.value(blk.fs.hxd.tube_outlet.flow_mol[0])))
        if use_surrogate:
            print(' Steam to charge (mol/s): {:.4f}'.format(
                pyo.value(blk.fs.steam_to_storage[0])))
            print(' Steam to discharge (mol/s): {:.4f}'.format(
                pyo.value(blk.fs.steam_to_discharge[0])))
        if not use_surrogate:
            print(' Makeup water flow: {:>.4f}'.format(
                pyo.value(blk.fs.condenser_mix.makeup.flow_mol[0])))
        print(' HXC temperature salt in/out (K): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.hxc.tube_inlet.temperature[0]),
            pyo.value(blk.fs.hxc.tube_outlet.temperature[0])))
        print(' HXD temperature salt in/out (K): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.hxd.shell_inlet.temperature[0]),
            pyo.value(blk.fs.hxd.shell_outlet.temperature[0])))
        print(' HXC delta temperature in/out (K): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.hxc.delta_temperature_in[0]),
            pyo.value(blk.fs.hxc.delta_temperature_out[0])))
        print(' HXD delta temperature in/out (K): {:.4f}/{:.4f}'.format(
            pyo.value(blk.fs.hxd.delta_temperature_in[0]),
            pyo.value(blk.fs.hxd.delta_temperature_out[0])))
        print(' HXC area (m2): {:.4f}'.format(pyo.value(blk.fs.hxc.area)))
        print(' HXD area (m2): {:.4f}'.format(pyo.value(blk.fs.hxd.area)))
        print(' HXC overall_heat_transfer_coefficient (m2): {:.4f}'.format(
            pyo.value(blk.fs.hxc.overall_heat_transfer_coefficient[0])))
        print(' HXD overall_heat_transfer_coefficient (m2): {:.4f}'.format(
            pyo.value(blk.fs.hxd.overall_heat_transfer_coefficient[0])))
        print(' HXC duty (MW): {:.4f}'.format(pyo.value(blk.fs.hxc.heat_duty[0])*1e-6))
        print(' HXD duty (MW): {:.4f}'.format(pyo.value(blk.fs.hxd.heat_duty[0])*1e-6))
        if not use_surrogate:
            print(' Split fraction to HXC: {:.4f}'.format(
                pyo.value(blk.fs.ess_charge_split.split_fraction[0, "to_hxc"])))
            print(' Split fraction to HXD: {:.4f}'.format(
                pyo.value(blk.fs.ess_discharge_split.split_fraction[0, "to_hxd"])))
        print(' Density mass in/out {:.4f}/{:.4f}'.format(
            pyo.value(m.period[1].fs.hxc.cold_side.properties_in[0].dens_mass["Liq"]),
            pyo.value(m.period[1].fs.hxc.cold_side.properties_out[0].dens_mass["Liq"])))
        c += 1

    print(results)


def plot_results(m,
                 blks,
                 lmp,
                 nweeks=None,
                 n_time_points=None,
                 net_power=None,
                 total_inventory=None,
                 hot_tank_level=None,
                 cold_tank_level=None,
                 hxc_duty=None,
                 hxd_duty=None,
                 boiler_heat_duty=None,
                 plant_heat_duty=None,
                 discharge_work=None,
                 pmax=None):


    # List of colors to be used in the plots
    c = ['darkred', 'navy', 'tab:green', 'k', 'gray']

    if n_time_points <= 50:
        marker_size = 8
        step_size = 2
    elif n_time_points <= 100:
        marker_size = 6
        step_size = 4
    else:
        marker_size = 3
        step_size = 12

    # Save and convert array to list to include values at time zero
    # for all the data that is going to be plotted
    hours = np.arange(n_time_points)
    lmp_array = np.asarray(lmp[0:n_time_points])
    hot_tank_array = np.asarray(hot_tank_level[0:n_time_points]).flatten()
    cold_tank_array = np.asarray(cold_tank_level[0:n_time_points]).flatten()
    hours_list = hours.tolist() + [n_time_points]
    hot_tank_list = [pyo.value(m.period[1].fs.previous_salt_inventory_hot)] + hot_tank_array.tolist()
    cold_tank_list = [pyo.value(m.period[1].fs.previous_salt_inventory_cold)] + cold_tank_array.tolist()
    hxc_array = np.asarray(hxc_duty[0:n_time_points]).flatten()
    hxd_array = np.asarray(hxd_duty[0:n_time_points]).flatten()
    hxc_duty_list = [0] + hxc_array.tolist()
    hxd_duty_list = [0] + hxd_array.tolist()
    if not use_surrogate:
        boiler_heat_duty_array = np.asarray(boiler_heat_duty[0:n_time_points]).flatten()
        boiler_heat_duty_list = [0] + boiler_heat_duty_array.tolist()
    plant_heat_duty_array = np.asarray(plant_heat_duty[0:n_time_points]).flatten()
    plant_heat_duty_list = [0] + plant_heat_duty_array.tolist()
    power_array = np.asarray(net_power[0:n_time_points]).flatten()
    power_list = [pyo.value(m.period[1].fs.previous_power)] + power_array.tolist()
    discharge_work_array = np.asarray(discharge_work[0:n_time_points]).flatten()
    discharge_work_list = [0] + discharge_work_array.tolist()

    # Plot salt inventory profiles
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xticks(range(0, n_time_points + 1, 1))
    ax1.set_xlabel('Time Period (hr)')
    ax1.set_ylabel('Salt Amount (metric ton)', color=c[3])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    # ax1.set_ylim((0, 2600))
    ax1.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    plt.axhline(pyo.value(m.period[1].fs.salt_amount), ls=':', lw=1.5, color=c[4], alpha=0.25)
    ax1.step(hours_list, hot_tank_list, marker='o', ms=marker_size, lw=1.5, color=c[0], alpha=0.85,
             label='Hot Salt')
    ax1.fill_between(hours_list, hot_tank_list, step="pre", color=c[0], alpha=0.25)
    ax1.step(hours_list, cold_tank_list, marker='o', ms=marker_size, lw=1.5, color=c[1], alpha=0.65,
             label='Cold Salt')
    ax1.fill_between(hours_list, cold_tank_list, step="pre", color=c[1], alpha=0.1)
    ax1.legend(loc="upper center", ncol=2, frameon=False)
    ax1.tick_params(axis='y')
    ax1.set_xticks(np.arange(0, n_time_points + 1, step=step_size))
    ax2 = ax1.twinx()
    ax2.set_ylim((0, 45))
    ax2.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax2.step([x + 1 for x in hours], lmp_array, marker='o', ms=marker_size,
             alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax2.tick_params(axis='y', labelcolor=c[2])
    if use_surrogate:
        plt.savefig('results/nlp_mp/surrogate_salt_tank_level_{}hrs.png'.format(n_time_points))
    else:
        plt.savefig('results/nlp_mp/salt_tank_level_{}hrs.png'.format(n_time_points))

    # Plot boiler and charge and discharge heat exchangers heat duty
    fig2, ax3 = plt.subplots(figsize=(12, 8))
    ax3.set_xlabel('Time Period (hr)')
    ax3.set_ylabel('Heat Duty (MW)', color=c[3])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(linestyle=':', which='both', color='gray', alpha=0.40)
    ax3.set_ylim((0, 200))
    # plt.axhline(pyo.value(max_storage_duty), ls=':', lw=1.5, color=c[4])
    plt.axhline(pyo.value(min_storage_duty), ls=':', lw=1.5, color=c[4])
    # plt.axhline(max(hxc_duty_list)*1.1, ls=':', lw=1.5, color=c[4])
    ax3.step(hours_list, hxc_duty_list, marker='o', ms=marker_size, color=c[0], alpha=0.75,
             label='Charge')
    ax3.fill_between(hours_list, hxc_duty_list, step="pre", color=c[0], alpha=0.25)
    ax3.step(hours_list, hxd_duty_list, marker='o', ms=marker_size, color=c[1], alpha=0.75,
             label='Discharge')
    ax3.fill_between(hours_list, hxd_duty_list, step="pre", color=c[1], alpha=0.1)
    ax3.tick_params(axis='y', labelcolor=c[3])
    ax3.legend(loc="upper center", ncol=2, frameon=False)
    ax3.tick_params(axis='y')
    ax3.set_xticks(np.arange(0, n_time_points + 1, step=step_size))
    ax4 = ax3.twinx()
    ax4.set_ylim((0, 45))
    ax4.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax4.step([x + 1 for x in hours], lmp_array, marker='o', ms=marker_size, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax4.tick_params(axis='y', labelcolor=c[2])
    if use_surrogate:
        plt.savefig('results/nlp_mp/surrogate_hx_heat_duty_{}hrs.png'.format(n_time_points))
    else:
        plt.savefig('results/nlp_mp/hx_heat_duty_{}hrs.png'.format(n_time_points))

    # Plot net power and discharge power profiles
    fig3, ax5 = plt.subplots(figsize=(12, 8))
    ax5.set_xlabel('Time Period (hr)')
    ax5.set_ylabel('Power Output (MW)', color='midnightblue')
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.grid(linestyle=':', which='both', color=c[4], alpha=0.40)
    ax5.set_ylim((0, 550))
    plt.axhline(pyo.value(pmax), ls=':', lw=1.5, color=c[4])
    ax5.step(hours_list, power_list, marker='o', ms=marker_size, lw=1.5, color=c[3], alpha=0.85,
             label='Net Power')
    ax5.fill_between(hours_list, power_list, step="pre", color=c[3], alpha=0.15)
    ax5.step(hours_list, discharge_work_list, marker='o', ms=marker_size, color=c[1], alpha=0.75,
             label='Storage Turbine')
    ax5.fill_between(hours_list, discharge_work_list, step="pre", color=c[1], alpha=0.15)
    ax5.tick_params(axis='y', labelcolor=c[1])
    ax5.legend(loc="upper center", ncol=2, frameon=False)
    ax5.set_xticks(np.arange(0, n_time_points + 1, step=step_size))
    ax6 = ax5.twinx()
    ax6.set_ylim((0, 45))
    ax6.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax6.step([x + 1 for x in hours], lmp_array, marker='o', ms=marker_size, alpha=0.7, ls='-', lw=1.5,
             color=c[2])
    ax6.tick_params(axis='y', labelcolor=c[2])
    if use_surrogate:
        plt.savefig('results/nlp_mp/surrogate_power_{}hrs.png'.format(n_time_points))
    else:
        plt.savefig('results/nlp_mp/power_{}hrs.png'.format(n_time_points))

    # Plot boiler and charge and discharge heat exchangers heat duty
    fig4, ax7 = plt.subplots(figsize=(12, 8))
    ax7.set_xlabel('Time Period (hr)')
    ax7.set_ylabel('Heat Duty (MW)', color=c[3])
    ax7.spines["top"].set_visible(False)
    ax7.spines["right"].set_visible(False)
    ax7.grid(linestyle=':', which='both', color='gray', alpha=0.40)
    ax7.step(hours_list, plant_heat_duty_list, marker='o', ms=marker_size, color=c[3], ls='-', lw=1.5, alpha=0.85,
             label='Plant')
    ax7.fill_between(hours_list, plant_heat_duty_list, step="pre", color=c[3], alpha=0.15)
    if not use_surrogate:
        ax7.step(hours_list, boiler_heat_duty_list, marker='o', ms=marker_size, color='gray', ls='-', lw=1.5, alpha=0.85, label='Boiler')
    ax7.tick_params(axis='y', labelcolor=c[3])
    ax7.legend(loc="upper center", ncol=2, frameon=False)
    ax7.tick_params(axis='y')
    ax7.set_xticks(np.arange(0, n_time_points + 1, step=step_size))
    ax8 = ax7.twinx()
    ax8.set_ylim((0, 45))
    ax8.set_ylabel('Locational Marginal Price ($/MWh)', color=c[2])
    ax8.step([x + 1 for x in hours], lmp_array, marker='o', ms=marker_size, alpha=0.7, ls='-', lw=1.5, color=c[2])
    ax8.tick_params(axis='y', labelcolor=c[2])
    if use_surrogate:
        plt.savefig('results/nlp_mp/surrogate_plant_heat_duty_{}hrs.png'.format(n_time_points))
    else:
        plt.savefig('results/nlp_mp/plant_heat_duty_{}hrs.png'.format(n_time_points))

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
        "max_iter": 150,
        # "halt_on_ampl_error": "yes",
    }
    solver = get_solver('ipopt', optarg)

    lx = False
    if lx:
        if use_surrogate:
            scaling_obj = 1e-1
        else:
            scaling_obj = 1e-1
    else:
        scaling_obj = 1e-1
    print()
    print('scaling_obj:', scaling_obj)

    # Add design data from .json file
    data_path = 'fixed_uscp_design_data.json'
    with open(data_path) as design_data:
        design_data_dict = json.load(design_data)

        max_salt_amount = pyo.units.convert(design_data_dict["max_salt_amount"]*pyunits.kg,
                                            to_units=pyunits.metric_ton)
        min_storage_duty = design_data_dict["min_storage_duty"]*pyunits.MW
        max_storage_duty = design_data_dict["max_storage_duty"]*pyunits.MW
        pmin_storage = design_data_dict["min_discharge_turbine_power"]*pyunits.MW
        pmax_storage = design_data_dict["max_discharge_turbine_power"]*pyunits.MW
        pmin = design_data_dict["plant_min_power"]*pyunits.MW
        pmax = design_data_dict["plant_max_power"]*pyunits.MW + pmax_storage

    hours_per_day = 24
    ndays = 1
    nhours = hours_per_day*ndays
    nweeks = 1

    # Add number of hours per week
    n_time_points = nweeks*nhours
    tank_status = "hot_empty"

    # Create a directory to save the results for each NLP sbproblem
    # and plots
    _mkdir('results')
    _mkdir('results/nlp_mp')

    # m = run_pricetaker_analysis(nweeks=nweeks,
    #                             n_time_points=n_time_points,
    #                             pmin=pmin,
    #                             pmax=pmax,
    #                             tank_status=tank_status,
    #                             max_salt_amount=max_salt_amount,
    #                             constant_salt=constant_salt)
    (m, blks, lmp, net_power, results, total_inventory,
     hot_tank_level, cold_tank_level, hxc_duty, hxd_duty,
     boiler_heat_duty,
     plant_heat_duty, discharge_work) = run_pricetaker_analysis(nweeks=nweeks,
                                                                n_time_points=n_time_points,
                                                                pmin=pmin,
                                                                pmax=pmax,
                                                                tank_status=tank_status,
                                                                max_salt_amount=max_salt_amount,
                                                                constant_salt=constant_salt)

    print_results(m, blks, results)

    plot_results(m, blks, lmp, nweeks=nweeks, n_time_points=n_time_points,
                 hot_tank_level=hot_tank_level, cold_tank_level=cold_tank_level,
                 net_power=net_power, hxc_duty=hxc_duty, hxd_duty=hxd_duty,
                 total_inventory=total_inventory,
                 boiler_heat_duty=boiler_heat_duty,
                 plant_heat_duty=plant_heat_duty, discharge_work=discharge_work, pmax=pmax)
