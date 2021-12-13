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
Simulation of Concrete TES units.
Author: Andres J. Calderon, Jaffer Ghouse
"""

import enum
from pandas.core.indexes import multi
from pyomo.core.base.expression import Expression
import pytest
from pyomo.environ import (ConcreteModel, TerminationCondition, Constraint, Objective, Var, RangeSet,
                            NonNegativeReals, SolverStatus, value, units as pyunits, SolverFactory, minimize)
from pyomo.util.calc_var_value import calculate_variable_from_constraint

from idaes.core import FlowsheetBlock, unit_model
from heat_exchanger_tube import ConcreteTubeSide
from idaes.generic_models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern

from idaes.generic_models.properties import iapws95

from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.util import get_solver
from idaes.core.util.testing import initialization_tester
from concrete_tes_concrete_side import add_1D_heat_transfer_equation, add_htc_surrogates
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
import pandas as pd
import math

# from supercritical_powerplant_simple_charge import initialize
import logging

from pyomo.util.infeasible import (log_infeasible_constraints,log_close_to_bounds)
logging.basicConfig(level=logging.INFO)

df_data = pd.read_excel(r'C:\Users\...\DISPATCHES\input_output_TES_data_500_samples.xlsx', sheet_name = "processed_data",
                                header=[0,1], index_col=None, keep_default_na=False)
number_samples = df_data.shape[0]
segments = 20
time_periods = 2
delta_time = 1800

multi_index = pd.MultiIndex.from_product([[i for i in range(number_samples)], [i for i in range(1,time_periods+1)]], names=['sample', 'time'])
df_concrete_Tprofiles = pd.DataFrame(index=multi_index, columns=list(range(1,segments+1)))
df_fluid_Tprofiles = pd.DataFrame(index=multi_index, columns=list(range(1,segments+1)))
df_vapor_fraction = pd.DataFrame(index=multi_index, columns=list(range(1,segments+1)))
df_U = pd.DataFrame(index=multi_index, columns=list(range(1,segments+1)))
solution_status = pd.DataFrame(index=[i for i in range(number_samples)], columns=['Surrogate','Unit init.', 'Square problem1', 'Simulation','Obj. function'])

m = ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": False})

m.fs.properties = iapws95.Iapws95ParameterBlock(default={
    "phase_presentation": iapws95.PhaseType.LG})

m.fs.error = Var(within=NonNegativeReals, bounds=(0, 1000000), initialize=5000)
m.fs.constant_htc = Var(within=NonNegativeReals, bounds=(20,500), initialize=100)
m.fs.objective = Objective(expr= m.fs.error, sense=minimize, doc="Objective function")


m.fs.time_periods = RangeSet(time_periods)


m.fs.unit= ConcreteTubeSide(m.fs.time_periods,
        default={"property_package": m.fs.properties,
                "flow_type": HeatExchangerFlowPattern.cocurrent,
                "transformation_method": "dae.finite_difference",
                "transformation_scheme": "BACKWARD",
                "has_pressure_change": True,
                "finite_elements": 19})

for t in m.fs.time_periods:

    unit = m.fs.unit[t]
    add_1D_heat_transfer_equation(unit)

    def temp_wall_constraint_rule(b,t,s):
        return b.temperature_wall[t, s] == b.surrogate.temperature_wall[t, s]
    unit.temp_wall_constraint = Constraint(unit.temperature_wall_index, rule=temp_wall_constraint_rule)

    # Setting deltaP =0 and adding bounds
    for i in unit.temperature_wall_index:
        unit.tube.deltaP[i].fix(0)
        unit.tube.properties[i].enth_mol.setlb(3000)
        unit.tube.properties[i].enth_mol.value = 50000
        unit.tube.properties[i].pressure.setub(30000000)
        unit.tube.properties[i].pressure.setlb(10000000)
        unit.tube.properties[i].pressure.value = 20000000
        unit.temperature_wall[i].setlb(300)
        unit.temperature_wall[i].setub(900)
        unit.temperature_wall[i].value = 400
        unit.tube.heat[i].value = -200


# for s in range(number_samples):
for s in [0]:
    # Unit = 91, fs = 71
    tfluid_inlet = df_data.loc[s,('T_fluid_sim_end')][1]
    pfluid_inlet = df_data.loc[s,('P_inlet')][0]
    tube_lenght = df_data.loc[s,('tube_length')][0]
    tube_diameter = df_data.loc[s,('tube_od')][0]
    flow_mol = df_data.loc[s,('mdot')][0]*1000/18.01528
    concrete_area = df_data.loc[s,('face_area')][0]

    T_concrete_init = list(df_data.loc[s,('T_concrete_sim_init')])
    # Concrete end temperature after ONE hour of operation.
    T_concrete_end = list(df_data.loc[s,('T_concrete_sim_end')])

    # Calculating an estimation of the final concrete temperature in case the simulation considers more than one hour
    for i in range(len(T_concrete_end)):
        T_concrete_end[i] = T_concrete_init[i] + (T_concrete_end[i] - T_concrete_init[i])*(delta_time*time_periods/3600)

    T_fluid_end = list(df_data.loc[s,('T_fluid_sim_end')])

    # The final wall temperature should be divided into intervals according to the number of time steps. i.e:
    # final temperature per segment = initial temperature + current time period * (final temperture - initial temperature)/(time periods)
    # This is an approximation to initialize the concrete side of each time block
    T_concrete_end_time = {}
    T_concrete_end_delta = []
    for t in m.fs.time_periods:
        for idx, i in enumerate(T_concrete_end):
            T_concrete_end_delta.append(T_concrete_init[idx] + (t)*(T_concrete_end[idx] - T_concrete_init[idx])/(time_periods))
        T_concrete_end_time.update({t:T_concrete_end_delta})
        T_concrete_end_delta = []

    # Unit = 71, fs = 142
    for t in m.fs.time_periods:
        unit = m.fs.unit[t]

        unit.tube_length.fix(tube_lenght)
        unit.d_tube_outer.fix(tube_diameter)
        unit.d_tube_inner.fix(unit.d_tube_outer.value)
        unit.tube_inlet.flow_mol[0].fix(flow_mol)  # mol/s
        unit.tube_inlet.pressure[0].fix(pfluid_inlet)  # Pa

        unit.surrogate.concrete_kappa.fix(1)
        unit.surrogate.delta_time.fix(delta_time)
        unit.surrogate.concrete_density.fix(2240)
        unit.surrogate.concrete_specific_heat.fix(900)
        unit.surrogate.concrete_face_area.fix(concrete_area)
        unit.surrogate.delta_z.fix(unit.tube_length.value/(segments-1))
        # Unit = 61, fs = 132,Unit = 61, fs = 122       

    # ********************************************************
    #        INITIALIZING HEAT TRANSFER SURROGATE MODEL
    # ********************************************************
    for i in m.fs.unit[m.fs.time_periods.first()].segments_set:
        m.fs.unit[m.fs.time_periods.first()].surrogate.wall_t_init[i].value = T_concrete_init[i-1]

    if not m.fs.find_component('heat_transfer_coefficient_surrogate'):
        add_htc_surrogates(m.fs)

    calculate_variable_from_constraint(m.fs.heat_transfer_coefficient_surrogate.mean_concrete_init_temp,
                                        m.fs.heat_transfer_coefficient_surrogate.mean_concrete_init_temp_constraint)
    calculate_variable_from_constraint(m.fs.constant_htc,
                                        m.fs.heat_transfer_coefficient_surrogate.htc_surrogate)

    # ********************************************************
    #        INITIALIZING EACH TIME BLOCK SEQUENTIALLY
    # ********************************************************
    for t in m.fs.time_periods:
        unit = m.fs.unit[t]

        unit.tube_inlet.enth_mol[0].fix(iapws95.htpx(tfluid_inlet*pyunits.K, pfluid_inlet*pyunits.Pa))  # K

        # Fixing initial concrete wall temperature and U coefficients
        for idx, i in enumerate(unit.temperature_wall_index):
            unit.surrogate.wall_t_init[idx+1].fix(T_concrete_init[idx])
            unit.tube_heat_transfer_coefficient[i].fix(value(m.fs.constant_htc))

        # Unit = 20, fs = 81, Unit = 20, fS = 20
        print(degrees_of_freedom(unit))

        # Fixing final wall temperature in order to initialize the concrete side
        for idx, i in enumerate(unit.temperature_wall_index):
            unit.surrogate.temperature_wall[i].fix(T_concrete_end_time[t][idx])

        iscale.set_scaling_factor(unit.tube.area, 1e-2)
        iscale.set_scaling_factor(unit.tube.heat, 1e-2)
        iscale.set_scaling_factor(unit.tube_heat_transfer_coefficient, 1e-2)
        iscale.calculate_scaling_factors(unit)
        # Unit = 0, fs = 61, Unit = 0, fS = 0
        assert degrees_of_freedom(unit) == 0

        # Get default solver for testing
        solver = get_solver()
        solver.options = {
                    "tol": 1e-6,
                    "max_iter": 100,
                    "halt_on_ampl_error": "yes",
                    # "bound_push": 1e-1,
                    # "mu_init": 1e-5
                }

        # **************************************************
        #            INITIALIZING CONCRETE SIDE  
        # **************************************************
        print("="*90)
        print(" "*10,"Initializing concrete side for Sample:",s,", time block",t," "*10)
        print("="*90)
        res = solver.solve(unit.surrogate, tee=True)
        solution_status.loc[s,'Surrogate'] = res.solver.termination_condition

        # **************************************************
        #               INITIALIZING TUBE SIDE  
        # **************************************************
        print("="*90)
        print(" "*10,"Initializing tube side for Sample:",s,", time block:",t," "*10)
        print("="*90)
        # Fixing tube side final wall temperature in order to initialize the tube side
        for i in unit.temperature_wall_index:
            unit.temperature_wall[i].fix(unit.surrogate.temperature_wall[i].value)

        try:
            unit.temp_wall_constraint.deactivate()
            unit.initialize(outlvl=idaeslog.DEBUG, optarg=solver.options)
            # Unit = 0, fs = 61
        except:
            pass

        # **************************************************
        #               INITIALIZING TIME BLOCK  
        # **************************************************
        # Unfixing wall temperature on concrete and tube side and activating an equality constraint
        unit.surrogate.temperature_wall.unfix()
        unit.temperature_wall.unfix()
        unit.temp_wall_constraint.activate()
        # Unit = 20, fs = 81, Unit = 20, fs = 20

        # Adding an equality constraint that links the heat transfer from the tube side to the concrete side
        def eq_heat_rule(b,t,s):
            return b.surrogate.q_fluid[(t,s)] == - b.tube.heat[(t,s)]*b.surrogate.delta_z

        if unit.find_component('eq_heat'):
            unit.eq_heat.activate()
        else:
            unit.eq_heat = Constraint(unit.temperature_wall_index, rule=eq_heat_rule)
        # Unit = 0, fs = 61

        print("="*80)
        print(" "*10,"Initializing tube + concrete sides for sample:",s, ",and block:", t," "*10)
        print("="*80)
        try:
            res = solver.solve(unit, tee=True, symbolic_solver_labels=True)
            solution_status.loc[s,'Square problem1'] = res.solver.termination_condition

        except:
            pass

        T_concrete_init = []
        for i in unit.temperature_wall_index:
            T_concrete_init.append(value(unit.surrogate.temperature_wall[i]))

    # ************************************************************
    #     ADDING EQUALITY CONSTRAINTS TO LINK EACH TIME BLOCK
    # ************************************************************
    for t in m.fs.time_periods:
        if t!=m.fs.time_periods.first():
            # Initial concrete temperature for time blocks greater or equal to 2 is unfixed.
            # A constraint is added linking the initial concrete temperature for time block t 
            # with the final concrete temperature from time t-1
            m.fs.unit[t].surrogate.wall_t_init.unfix()

            for i in m.fs.unit[t].segments_set:
                if m.fs.find_component('eq_link_temp_block{}_segment_{}'.format(t,i)):
                    eq_link_temp = getattr(m.fs, 'eq_link_temp_block{}_segment_{}'.format(t,i))
                    eq_link_temp.activate()
                else:
                    eq_link_temp = Constraint(
                        expr=m.fs.unit[t].surrogate.wall_t_init[i] == m.fs.unit[t-1].temperature_wall[m.fs.unit[t].temperature_wall_index.ordered_data()[i-1]])
                    setattr(m.fs, 'eq_link_temp_block{}_segment_{}'.format(t,i),eq_link_temp)

    try:
        # Solving all time blocks simultaneously
        print("="*80)
        print(" "*10,"Solving", t, "time block(s) simultaneously for sample:",s," "*10)
        print("="*80)
        res = solver.solve(m.fs, tee=True, symbolic_solver_labels=True)
        log_close_to_bounds(m.fs)
        log_infeasible_constraints(m.fs)
        solution_status.loc[s, 'Simulation'] = res.solver.termination_condition
        solution_status.loc[s, 'Obj. function'] = value(m.fs.error)

        for t in m.fs.unit:
            for i in m.fs.unit[t].segments_set:
                df_concrete_Tprofiles.loc[(s,t),i] = value(m.fs.unit[t].surrogate.temperature_wall[m.fs.unit[t].temperature_wall_index.ordered_data()[i-1]])
                df_fluid_Tprofiles.loc[(s,t),i] = value(m.fs.unit[t].tube.properties[m.fs.unit[t].temperature_wall_index.ordered_data()[i-1]].temperature)
                df_vapor_fraction.loc[(s,t),i] = value(m.fs.unit[t].tube.properties[m.fs.unit[t].temperature_wall_index.ordered_data()[i-1]].vapor_frac)
                df_U.loc[(s,t),i] = value(m.fs.unit[t].tube_heat_transfer_coefficient[m.fs.unit[t].temperature_wall_index.ordered_data()[i-1]])

    except:
        pass

    for t in m.fs.unit:
        m.fs.unit[t].eq_heat.deactivate()
        if t!=m.fs.time_periods.first():
            for i in m.fs.unit[t].segments_set:
                eq_link_temp = getattr(m.fs, 'eq_link_temp_block{}_segment_{}'.format(t,i))
                eq_link_temp.deactivate()

fname = "Case 2 Multiperiod Concrete TES Surrogate Us.xlsx"
with pd.ExcelWriter(fname) as writer:
    df_fluid_Tprofiles.to_excel(writer, sheet_name="Fluid Temp. profiles")
    df_concrete_Tprofiles.to_excel(writer, sheet_name="Concrete Temp. profiles")
    df_vapor_fraction.to_excel(writer, sheet_name="Vapor fraction profiles")
    df_U.to_excel(writer, sheet_name="Heat transfer coefficients")
    solution_status.to_excel(writer, sheet_name='Solver status')
