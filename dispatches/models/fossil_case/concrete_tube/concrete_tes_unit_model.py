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
Concrete TES model.
Author: Andres J. Calderon, Jaffer Ghouse
"""

from pyomo.core.base.block import Block
from pyomo.environ import (ConcreteModel, TerminationCondition, Constraint, Var, RangeSet,
                            NonNegativeReals, value, units as pyunits, minimize)
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core import FlowsheetBlock
from heat_exchanger_tube import ConcreteTubeSide
from idaes.generic_models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern

from idaes.generic_models.properties import iapws95

from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.util import get_solver
from concrete_tes_concrete_side import add_1D_heat_transfer_equation, add_htc_surrogates, add_capex_surrogate
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
import pandas as pd
import math

import logging

from pyomo.util.infeasible import (log_infeasible_constraints,log_close_to_bounds)
logging.basicConfig(level=logging.INFO)

def add_concrete_tes(m, data=None):

    m.fs.concrete_tes = Block()
    m.fs.concrete_tes.time_periods = RangeSet(data['time_periods'])
    m.fs.concrete_tes.number_tubes = Var(within=NonNegativeReals, bounds=(1,100000), initialize=data["number_tubes"], doc='Number of tubes of the concrete TES')
    m.fs.concrete_tes.constant_htc = Var(within=NonNegativeReals, bounds=(20,500), initialize=100)
    m.fs.concrete_tes.number_tubes.fix()

    m.fs.concrete_tes.unit= ConcreteTubeSide(m.fs.concrete_tes.time_periods,
            default={"property_package": m.fs.prop_water,
                    "flow_type": HeatExchangerFlowPattern.cocurrent,
                    "transformation_method": "dae.finite_difference",
                    "transformation_scheme": "BACKWARD",
                    "has_pressure_change": True,
                    "finite_elements": data["segments"]-1})

    for t in m.fs.concrete_tes.time_periods:
        unit = m.fs.concrete_tes.unit[t]
        add_1D_heat_transfer_equation(unit)

        def temp_wall_constraint_rule(b,t,s):
            return b.temperature_wall[t, s] == b.surrogate.temperature_wall[t, s]
        unit.temp_wall_constraint = Constraint(unit.temperature_wall_index, rule=temp_wall_constraint_rule)

        # Setting deltaP =0 and adding bounds
        for i in unit.temperature_wall_index:
            unit.tube.deltaP[i].fix(data['deltaP'])
            unit.tube.properties[i].enth_mol.setlb(3000)
            unit.tube.properties[i].enth_mol.value = 50000
            unit.tube.properties[i].pressure.setub(30000000)
            unit.tube.properties[i].pressure.setlb(4000000)
            unit.tube.properties[i].pressure.value = 20000000
            unit.temperature_wall[i].setlb(300)
            unit.temperature_wall[i].setub(900)
            unit.temperature_wall[i].value = 400
            unit.tube.heat[i].value = -200

        # Fixing geometry of the tube side
        unit.tube_length.fix(data["tube_length"])
        unit.d_tube_outer.fix(data["tube_diameter"])
        unit.d_tube_inner.fix(unit.d_tube_outer.value)

        # Fixing deltaT for concrete side and concrete properties
        unit.surrogate.delta_time.fix(data["delta_time"])
        unit.surrogate.concrete_kappa.fix(data["concrete_difussivity"])
        unit.surrogate.concrete_density.fix(data["concrete_density"])
        unit.surrogate.concrete_specific_heat.fix(data["concrete_specific_heat"])
        unit.surrogate.concrete_face_area.fix(data["concrete_area"])
        unit.surrogate.delta_z.fix(unit.tube_length.value/(data["segments"]-1))

        # Fixing the initial concrete temperature for the first time block
        if t == m.fs.concrete_tes.time_periods.first():
            for i in unit.segments_set:
                unit.surrogate.wall_t_init[i].fix(data['concrete_init_temp'][i-1])

    # Adding surrogate for heat transfer coefficient and calculating its value
    add_htc_surrogates(m.fs.concrete_tes)
    calculate_variable_from_constraint(m.fs.concrete_tes.heat_transfer_coefficient_surrogate.mean_concrete_init_temp,
                                        m.fs.concrete_tes.heat_transfer_coefficient_surrogate.mean_concrete_init_temp_constraint)
    calculate_variable_from_constraint(m.fs.concrete_tes.constant_htc,
                                        m.fs.concrete_tes.heat_transfer_coefficient_surrogate.htc_surrogate)
    
    add_capex_surrogate(m.fs.concrete_tes)
    calculate_variable_from_constraint(m.fs.concrete_tes.capex,
                                        m.fs.concrete_tes.capex_surrogate)

    for t in m.fs.concrete_tes.time_periods:
        unit = m.fs.concrete_tes.unit[t]
        unit.tube_heat_transfer_coefficient.fix(value(m.fs.concrete_tes.constant_htc))


    iscale.set_scaling_factor(unit.tube.area, 1e-2)
    iscale.set_scaling_factor(unit.tube.heat, 1e-2)
    iscale.set_scaling_factor(unit.tube_heat_transfer_coefficient, 1e-2)
    iscale.calculate_scaling_factors(unit)

def initialize_concrete_tes(m, data=None):

    inlet_pressure = data["inlet_pressure"]
    flow_mol = data["flow_mol"]
    concrete_init_temp = data['concrete_init_temp']
    concrete_final_temp = data['concrete_final_temp']

    if "inlet_temperature" in data.keys():
        inlet_enthalpy = iapws95.htpx(data["inlet_temperature"]*pyunits.K, inlet_pressure*pyunits.Pa)
    
    elif "inlet_enthalpy" in data.keys():
        inlet_enthalpy = data["inlet_enthalpy"]
    
    else:
        raise Exception("The concrete TES Input data must contain either inlet_enthalpy or inlet_temperature. If both are provided, inlet_temperature is used")

    # The final wall temperature should be divided into intervals according to the number of time steps. i.e:
    # final temperature per segment = initial temperature + current time period * (final temperture - initial temperature)/(time periods)
    # This is an approximation to initialize the concrete side of each time block
    T_concrete_end_time = {}
    T_concrete_end_delta = []
    for t in m.fs.concrete_tes.time_periods:
        for idx, i in enumerate(concrete_final_temp):
            T_concrete_end_delta.append(concrete_init_temp[idx] + (t)*(concrete_final_temp[idx] - concrete_init_temp[idx])/m.fs.concrete_tes.time_periods.last())
        T_concrete_end_time.update({t:T_concrete_end_delta})
        T_concrete_end_delta = []

    # ********************************************************
    #        INITIALIZING EACH TIME BLOCK SEQUENTIALLY
    # ********************************************************
    for t in m.fs.concrete_tes.time_periods:
        unit = m.fs.concrete_tes.unit[t]

        unit.tube_inlet.flow_mol[0].fix(flow_mol)  # mol/s
        unit.tube_inlet.pressure[0].fix(inlet_pressure)  # Pa
        unit.tube_inlet.enth_mol[0].fix(inlet_enthalpy)
        # unit.tube_inlet.enth_mol[0].fix(inlet_enthalpy)  # K

        # Fixing initial concrete wall temperature
        for i in unit.segments_set:
            unit.surrogate.wall_t_init[i].fix(concrete_init_temp[i-1])

        print(degrees_of_freedom(unit))

        # Fixing final wall temperature in order to initialize the concrete side
        for idx, i in enumerate(unit.temperature_wall_index):
            unit.surrogate.temperature_wall[i].fix(T_concrete_end_time[t][idx])

        assert degrees_of_freedom(unit) == 0

        # Get default solver for testing
        solver = get_solver()
        solver.options = {
                    "tol": 1e-6,
                    "max_iter": 100,
                    "halt_on_ampl_error": "yes",
                    "bound_push": 1e-1,
                    # "mu_init": 1e-5
                }

        # **************************************************
        #            INITIALIZING CONCRETE SIDE  
        # **************************************************
        print("="*90)
        print(" "*20,"Initializing concrete side for time block:",t," "*20)
        print("="*90)
        res = solver.solve(unit.surrogate, tee=True)

        # **************************************************
        #               INITIALIZING TUBE SIDE 
        # **************************************************
        print()
        print("="*90)
        print(" "*20,"Initializing tube side for time block:",t," "*20)
        print("="*90)

        # Fixing tube side final wall temperature in order to initialize the tube side
        for i in unit.temperature_wall_index:
            unit.temperature_wall[i].fix(unit.surrogate.temperature_wall[i].value)

        try:
            unit.temp_wall_constraint.deactivate()
            unit.initialize(outlvl=idaeslog.DEBUG, optarg=solver.options)

        except:
            pass

        # **************************************************
        #               INITIALIZING TIME BLOCK  
        # **************************************************
        # Unfixing wall temperature on concrete and tube side and activating an equality constraint
        unit.surrogate.temperature_wall.unfix()
        unit.temperature_wall.unfix()
        unit.temp_wall_constraint.activate()

        # Adding an equality constraint that links the heat transfer from the tube side to the concrete side
        def eq_heat_rule(b,t,s):
            return b.surrogate.q_fluid[(t,s)] == - b.tube.heat[(t,s)]*b.surrogate.delta_z

        if unit.find_component('eq_heat'):
            unit.eq_heat.activate()
        else:
            unit.eq_heat = Constraint(unit.temperature_wall_index, rule=eq_heat_rule)
        
        print()
        print("="*90)
        print(" "*20,"Initializing tube + concrete sides for block:", t," "*20)
        print("="*90)
        try:
            res = solver.solve(unit, tee=True, symbolic_solver_labels=True)

        except:
            pass

        # The initial concrete temperature for the next time block, is equal to the final concrete
        # temperature for the current time block
        concrete_init_temp = []
        for i in unit.temperature_wall_index:
            concrete_init_temp.append(value(unit.surrogate.temperature_wall[i]))


    # ************************************************************
    #     ADDING EQUALITY CONSTRAINTS TO LINK EACH TIME BLOCK
    # ************************************************************
    for t in m.fs.concrete_tes.time_periods:
        if t != m.fs.concrete_tes.time_periods.first():
            # Initial concrete temperature for time blocks greater or equal to 2 is unfixed.
            # A constraint is added linking the initial concrete temperature for time block t 
            # with the final concrete temperature from time t-1
            m.fs.concrete_tes.unit[t].surrogate.wall_t_init.unfix()

            for i in m.fs.concrete_tes.unit[t].segments_set:
                if m.fs.concrete_tes.find_component('eq_link_temp_block{}_segment_{}'.format(t,i)):
                    eq_link_temp = getattr(m.fs.concrete_tes, 'eq_link_temp_block{}_segment_{}'.format(t,i))
                    eq_link_temp.activate()
                else:
                    eq_link_temp = Constraint(
                        expr=m.fs.concrete_tes.unit[t].surrogate.wall_t_init[i] == m.fs.concrete_tes.unit[t-1].temperature_wall[m.fs.concrete_tes.unit[t].temperature_wall_index.ordered_data()[i-1]])
                    setattr(m.fs.concrete_tes, 'eq_link_temp_block{}_segment_{}'.format(t,i),eq_link_temp)

    try:
        res = solver.solve(m.fs.concrete_tes, tee=True, symbolic_solver_labels=True)

    except:
        pass

    for t in m.fs.concrete_tes.time_periods:
        m.fs.concrete_tes.unit[t].tube_heat_transfer_coefficient.unfix()
        m.fs.concrete_tes.unit[t].tube_heat_transfer_coefficient.setlb(20)
        m.fs.concrete_tes.unit[t].tube_heat_transfer_coefficient.setub(400)

        # Adding equality constraints to ensure that U is the same for all the segments
        def eq_ht_coefficient_rule(b,t,s):
            return  b.tube_heat_transfer_coefficient[(t,s)] == m.fs.concrete_tes.constant_htc

        if m.fs.concrete_tes.unit[t].find_component('eq_heat_transfer_coefficient'):
            m.fs.concrete_tes.unit[t].eq_heat_transfer_coefficient.activate()
        else:
            m.fs.concrete_tes.unit[t].eq_heat_transfer_coefficient = Constraint(m.fs.concrete_tes.unit[t].temperature_wall_index, rule=eq_ht_coefficient_rule)

    try:
        res = solver.solve(m.fs.concrete_tes, tee=True,symbolic_solver_labels=True)
        log_close_to_bounds(m.fs.concrete_tes)
        log_infeasible_constraints(m.fs.concrete_tes)

    except:
        pass


if __name__ == "__main__":

    m = ConcreteModel(name="Steam Cycle Model")
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.prop_water = iapws95.Iapws95ParameterBlock()

    # Input data required to build the concrete TES
    data = {"delta_time": 1800,
            "time_periods": 2,
            "segments": 20,
            "deltaP": 0,
            "concrete_init_temp": [750, 732.631579, 715.2631579, 697.8947368, 680.5263158, 663.1578947,
                                645.7894737, 628.4210526, 611.0526316, 593.6842105, 576.3157895, 558.9473684,
                                541.5789474, 524.2105263, 506.8421053, 489.4736842, 472.1052632, 454.7368421,
                                437.3684211, 420],
            "tube_length": 64.9,
            "tube_diameter": 0.0105664,
            "number_tubes": 10000,
            "concrete_area": 0.00847,
            "concrete_difussivity": 1,
            "concrete_density": 2240,
            "concrete_specific_heat": 900,
            # This data is used in the initialization step, concrete_final_temp is recalculated during the initialization step
            "concrete_final_temp": [787.049044, 768.2461577, 749.7581953, 731.6762425, 713.9605891, 696.5867905,
                                    679.5450987, 662.8403567, 646.4919514, 630.5145062, 614.9422052, 599.8251406,
                                    585.2301277, 571.242303, 557.9698795, 545.6035881, 534.0878954, 523.0092146,
                                    511.8313733, 500.2123965],
            "flow_mol": 0.00958*1000/18.01528*0+0.2911100000000000,
            "inlet_pressure": 19600000*0+24235081.4,
            "inlet_temperature": 853.92699435*0+866.1500004837724,
            }

    # The TES is added to the flowsheet
    add_concrete_tes(m, data)

    # Initialize the unit model
    initialize_concrete_tes(m, data)

    print(degrees_of_freedom(m.fs.concrete_tes))

    df_concrete_Tprofiles = pd.DataFrame(index=list(m.fs.concrete_tes.time_periods), columns=list(range(1,data["segments"]+1)))
    df_fluid_Tprofiles = pd.DataFrame(index=list(m.fs.concrete_tes.time_periods), columns=list(range(1,data["segments"]+1)))
    df_vapor_fraction = pd.DataFrame(index=list(m.fs.concrete_tes.time_periods), columns=list(range(1,data["segments"]+1)))
    df_U = pd.DataFrame(index=list(m.fs.concrete_tes.time_periods), columns=list(range(1,data["segments"]+1)))

    for t in m.fs.concrete_tes.time_periods:
        for idx, i in enumerate(m.fs.concrete_tes.unit[t].temperature_wall_index):
            df_concrete_Tprofiles.loc[(t),idx+1] = value(m.fs.concrete_tes.unit[t].surrogate.temperature_wall[i])
            df_fluid_Tprofiles.loc[(t),idx+1] = value(m.fs.concrete_tes.unit[t].tube.properties[i].temperature)
            df_vapor_fraction.loc[(t),idx+1] = value(m.fs.concrete_tes.unit[t].tube.properties[i].vapor_frac)
            df_U.loc[(t),idx+1] = value(m.fs.concrete_tes.unit[t].tube_heat_transfer_coefficient[i])

