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
"""
TES surrogates for concrete wall temperature.
Heating source: steam
Thermal material: Concrete
Author: Andres J Calderon, Jaffer Ghouse, Storeworks
Date: July 19, 2021
"""

# Import Pyomo libraries
from pyomo.core.base.block import Block
import pyomo.environ as pyo
from idaes.core import FlowsheetBlock
from pyomo.environ import log, exp
from math import pi
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.core.util.scaling as iscale

def add_1D_heat_transfer_equation(b):

    b.surrogate = pyo.ConcreteModel()
    b.segments_set = pyo.RangeSet(20)
    b.surrogate.concrete_kappa = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,10), initialize=1, doc='Conductivity constant [J/m.K.s]')
    b.surrogate.delta_time = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,4000), initialize=3600, doc='Delta for discretizing operation time [s]')
    b.surrogate.concrete_density = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,3000), initialize=2240, doc='Concrete density [kg/m3]')
    b.surrogate.concrete_specific_heat = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,1000), initialize=900, doc='Concrete specific heat [J/kg.K]')
    b.surrogate.concrete_face_area  =  pyo.Var(within=pyo.NonNegativeReals, bounds=(0.003,0.015), initialize=0.01, doc='Face area of the concrete wall [m2]')
    b.surrogate.delta_z =  pyo.Var(within=pyo.NonNegativeReals, bounds=(0,100), initialize=5, doc='Delta for discretizing tube length [m]')
    b.surrogate.q_fluid = pyo.Var(b.temperature_wall_index, within=pyo.NonNegativeReals, bounds=(-5000,5000), initialize=600, doc='Q transferred from the steam to the concrete segment [J/s]')
    b.surrogate.wall_t_init=  pyo.Var(b.segments_set, within=pyo.NonNegativeReals, bounds=(300,900), initialize=600, doc='Initial concrete wall temperature [K]')
    b.surrogate.temperature_wall  =  pyo.Var(b.temperature_wall_index, within=pyo.NonNegativeReals, bounds=(300,900), initialize=600, doc='Final concrete wall temperature [K]')

    b.surrogate.temp_segment1_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[0]] == b.surrogate.wall_t_init[1] - \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[1] - b.surrogate.wall_t_init[2]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[0]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment2_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[1]] == b.surrogate.wall_t_init[2] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[3] - 2*b.surrogate.wall_t_init[2] + b.surrogate.wall_t_init[1]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[1]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment3_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[2]] == b.surrogate.wall_t_init[3] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[4] - 2*b.surrogate.wall_t_init[3] + b.surrogate.wall_t_init[2]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[2]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment4_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[3]] == b.surrogate.wall_t_init[4] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[5] - 2*b.surrogate.wall_t_init[4] + b.surrogate.wall_t_init[3]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[3]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment5_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[4]] == b.surrogate.wall_t_init[5] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[6] - 2*b.surrogate.wall_t_init[5] + b.surrogate.wall_t_init[4]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[4]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment6_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[5]] == b.surrogate.wall_t_init[6] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[7] - 2*b.surrogate.wall_t_init[6] + b.surrogate.wall_t_init[5]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[5]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment7_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[6]] == b.surrogate.wall_t_init[7] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[8] - 2*b.surrogate.wall_t_init[7] + b.surrogate.wall_t_init[6]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[6]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment8_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[7]] == b.surrogate.wall_t_init[8] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[9] - 2*b.surrogate.wall_t_init[8] + b.surrogate.wall_t_init[7]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[7]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment9_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[8]] == b.surrogate.wall_t_init[9] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[10] - 2*b.surrogate.wall_t_init[9] + b.surrogate.wall_t_init[8]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[8]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment10_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[9]] == b.surrogate.wall_t_init[10] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[11] - 2*b.surrogate.wall_t_init[10] + b.surrogate.wall_t_init[9]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[9]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment11_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[10]] == b.surrogate.wall_t_init[11] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[12] - 2*b.surrogate.wall_t_init[11] + b.surrogate.wall_t_init[10]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[10]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment12_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[11]] == b.surrogate.wall_t_init[12] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[13] - 2*b.surrogate.wall_t_init[12] + b.surrogate.wall_t_init[11]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[11]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment13_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[12]] == b.surrogate.wall_t_init[13] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[14] - 2*b.surrogate.wall_t_init[13] + b.surrogate.wall_t_init[12]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[12]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment14_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[13]] == b.surrogate.wall_t_init[14] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[15] - 2*b.surrogate.wall_t_init[14] + b.surrogate.wall_t_init[13]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[13]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment15_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[14]] == b.surrogate.wall_t_init[15] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[16] - 2*b.surrogate.wall_t_init[15] + b.surrogate.wall_t_init[14]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[14]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment16_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[15]] == b.surrogate.wall_t_init[16] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[17] - 2*b.surrogate.wall_t_init[16] + b.surrogate.wall_t_init[15]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[15]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment17_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[16]] == b.surrogate.wall_t_init[17] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[18] - 2*b.surrogate.wall_t_init[17] + b.surrogate.wall_t_init[16]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[16]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment18_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[17]] == b.surrogate.wall_t_init[18] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[19] - 2*b.surrogate.wall_t_init[18] + b.surrogate.wall_t_init[17]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[17]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment19_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[18]] == b.surrogate.wall_t_init[19] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[20] - 2*b.surrogate.wall_t_init[19] + b.surrogate.wall_t_init[18]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[18]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

    b.surrogate.temp_segment20_constraint = pyo.Constraint(expr=b.surrogate.temperature_wall[b.temperature_wall_index.ordered_data()[19]] == b.surrogate.wall_t_init[20] + \
        b.surrogate.concrete_kappa*b.surrogate.delta_time/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.delta_z**2)*(b.surrogate.wall_t_init[19] - b.surrogate.wall_t_init[20]) + \
            b.surrogate.delta_time*b.surrogate.q_fluid[b.temperature_wall_index.ordered_data()[19]]/(b.surrogate.concrete_density*b.surrogate.concrete_specific_heat*b.surrogate.concrete_face_area*b.surrogate.delta_z))

def add_htc_surrogates(b):

    b.heat_transfer_coefficient_surrogate = pyo.ConcreteModel()

    b.heat_transfer_coefficient_surrogate.mean_concrete_init_temp = pyo.Var(within=pyo.NonNegativeReals, bounds=(300,1000), initialize=500, doc='Mean initial concrete wall temperature')

    for i in b.time_periods:
        if i == b.time_periods.first():
            def mean_concrete_init_temp_rule(b):
                c = b.parent_block()
                return b.mean_concrete_init_temp == sum(c.unit[i].surrogate.wall_t_init[s] for s in c.unit[i].segments_set)/c.unit[i].segments_set.last()
            b.heat_transfer_coefficient_surrogate.mean_concrete_init_temp_constraint = pyo.Constraint(rule=mean_concrete_init_temp_rule)

    for i in b.time_periods:
        if i == b.time_periods.first():

            # Heat_transfer_coefficient = f(concrete_face_area, d_tube_outer, tube_inlet.pressure)
            b.heat_transfer_coefficient_surrogate.htc_surrogate = pyo.Constraint(expr=b.constant_htc ==  -153.30491285800843570541 * b.unit[i].surrogate.concrete_face_area - 8386.3028048661253706086 * b.unit[i].d_tube_outer - 0.48582603845628324901185 * (b.unit[i].tube_inlet.pressure[0]/1000000) - 0.24273547980088984796351E-006 * b.unit[i].d_tube_outer**-4 - 0.15780375527247002417718E-005 * b.unit[i].surrogate.concrete_face_area**-3 + 0.90901584753593953850542E-003 * b.unit[i].surrogate.concrete_face_area**-2 + 179.56349723936409645830)

def add_capex_surrogate(b):

    b.capex = pyo.Var(within=pyo.NonNegativeReals, bounds=(0,2000), initialize=700, doc='Bare Erected Cost (BER) Per Tube [USD]')
    
    for i in b.time_periods:
        if i == b.time_periods.first():
            # Capex per tube = f(tube_length, concrete_area, tube_diamter, inlet_pressure)
            b.capex_surrogate = pyo.Constraint(expr=b.capex == - 5.0330935178250681971690 * b.unit[i].tube_length - 17482.067135713899915572 * b.unit[i].surrogate.concrete_face_area + 46204.834011172177270055 * b.unit[i].d_tube_outer \
                - 23.091759612009159496893 * (b.unit[i].tube_inlet.pressure[0]/1000000) - 0.15657040320651968794933E-005 * b.unit[i].surrogate.concrete_face_area**-3 - 0.15531449903948327771752E-003 * b.unit[i].d_tube_outer**-3 \
                    + 5860311.3183172149583697 * (b.unit[i].tube_inlet.pressure[0]/1000000)**-3 - 219928.20303379974211566 * b.unit[i].tube_length**-2 + 0.88773788864419283291957E-003 * b.unit[i].surrogate.concrete_face_area**-2 \
                        - 689783.54911381844431162 * (b.unit[i].tube_inlet.pressure[0]/1000000)**-2 + 6.6453732481704159695823 * b.unit[i].d_tube_outer**-1 - 1.6141255301306027813979 * (b.unit[i].d_tube_outer*(b.unit[i].tube_inlet.pressure[0]/1000000))**-3 \
                            - 0.22290734378486113154439E-007 * (b.unit[i].surrogate.concrete_face_area*b.unit[i].d_tube_outer)**-2 + 21.785281602999724270830 * (b.unit[i].d_tube_outer*(b.unit[i].tube_inlet.pressure[0]/1000000))**-2 \
                                + 209417.65877250320045277 * (b.unit[i].tube_length*(b.unit[i].tube_inlet.pressure[0]/1000000))**-1 + 883.99716857681733017671 * b.unit[i].tube_length*b.unit[i].surrogate.concrete_face_area \
                                    + 853.32799887986300291232 * b.unit[i].tube_length*b.unit[i].d_tube_outer + 1574960.0062072663567960 * b.unit[i].surrogate.concrete_face_area*b.unit[i].d_tube_outer)
