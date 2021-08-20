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
Tests for ConcreteTubeSide model.
Author: Konica Mulani, Jaffer Ghouse
"""

import pytest
from pyomo.environ import (ConcreteModel, TerminationCondition, Constraint, RangeSet,
                           SolverStatus, value, units as pyunits, SolverFactory)

from idaes.core import FlowsheetBlock
from heat_exchanger_tube import ConcreteTubeSide
from idaes.generic_models.unit_models.heat_exchanger \
    import HeatExchangerFlowPattern

from idaes.generic_models.properties import iapws95

from idaes.core.util.model_statistics import degrees_of_freedom

from idaes.core.util import get_solver
from idaes.core.util.testing import initialization_tester
from dispatches.models.fossil_case.concrete_tube.tes_surrogates_coupled import add_surrogates
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
import pandas as pd



m = ConcreteModel()
m.fs = FlowsheetBlock(default={"dynamic": False})

m.fs.properties = iapws95.Iapws95ParameterBlock(default={
    "phase_presentation": iapws95.PhaseType.LG})

m.fs.unit = ConcreteTubeSide(
    default={"property_package": m.fs.properties,
             "flow_type": HeatExchangerFlowPattern.cocurrent,
             "transformation_method": "dae.finite_difference",
             "transformation_scheme": "BACKWARD",
             "has_pressure_change": True,
             "finite_elements": 19})

add_surrogates(m.fs.unit)

df_test = pd.read_excel(r'C:\Users\jamey\OneDrive\Documents\KeyLogic\DISPATCHES\input_output_TES_data_500_samples.xlsx', sheet_name = "test_data",
                                header=[0,1], index_col=None, keep_default_na=False)

df_tfluid = pd.DataFrame(index = range(50), columns = list(m.fs.unit.segments_set.data()))
# tinit_concrete = [862, 842.05263158, 822.10526316, 802.15789474, 782.21052632, 762.26315789, 742.31578947, 722.36842105, 702.42105263, 682.47368421, 662.52631579, 642.57894737, 622.63157895, 602.68421053, 582.73684211, 562.78947368, 542.84210526, 522.89473684, 502.94736842, 483]

print(iapws95.htpx(298*pyunits.K, 17900000*pyunits.Pa))

# 873

for i in m.fs.unit.temperature_wall_index:
    eq_wall = Constraint(
        expr=m.fs.unit.temperature_wall[0, i[1]] ==
        m.fs.unit.surrogate.temperature_wall[0, i[1]])
    setattr(m.fs, 'eq_wall{}'.format(i[1]), eq_wall)

for i in m.fs.unit.temperature_wall_index:
    m.fs.unit.tube.deltaP[i].fix(1160)

# m.fs.unit.temperature_wall.setlb(300)
# m.fs.unit.temperature_wall.setub(900)
# for i in m.fs.unit.segments_set:
#     m.fs.unit.surrogate.wall_t_init[i].fix(tinit_concrete[i-1])
# m.fs.unit.surrogate.wall_t_init.display()
solution_state = {}
for s in [30]:
    for i in m.fs.unit.segments_set:
        m.fs.unit.surrogate.wall_t_init[i].fix(round(df_test.loc[s,('T_concrete_sim_init')][i],4))
    # m.fs.unit.surrogate.wall_t_init.display()
    # Variable created by the add_surrogate method that mirrors the variables from the 1D side tube model

    m.fs.unit.tube_length.fix(df_test.loc[s,('tube_length')][0])
    m.fs.unit.d_tube_outer.fix(df_test.loc[s,('tube_od')][0])
    m.fs.unit.d_tube_inner.fix(m.fs.unit.d_tube_outer.value)
    m.fs.unit.tube_inlet.flow_mol[0].fix(df_test.loc[s,('mdot')][0]*1000/18.01528)  # mol/s
    m.fs.unit.tube_inlet.pressure[0].fix(df_test.loc[s,('P_inlet')][0])  # Pa

    m.fs.unit.surrogate.tube_length.fix(m.fs.unit.tube_length.value)
    m.fs.unit.surrogate.face_area.fix(df_test.loc[s,('face_area')][0])
    m.fs.unit.surrogate.d_tube_outer.fix(m.fs.unit.d_tube_outer.value) 
    m.fs.unit.surrogate.flow_mol.fix(m.fs.unit.tube_inlet.flow_mol[0].value)
    m.fs.unit.surrogate.pressure.fix(m.fs.unit.tube_inlet.pressure[0].value)

    print(degrees_of_freedom(m.fs.unit))

    # m.fs.unit.surrogate.constraint_tube_length = Constraint(expr=m.fs.unit.surrogate.tube_length==m.fs.unit.tube_length)
    # m.fs.unit.surrogate.constraint_face_area = Constraint(expr=m.fs.unit.surrogate.face_area==df_test.loc[41,('face_area')][0])
    # m.fs.unit.surrogate.constraint_tube_od = Constraint(expr=m.fs.unit.surrogate.d_tube_outer==m.fs.unit.d_tube_outer)
    # m.fs.unit.surrogate.constraint_mdot = Constraint(expr=m.fs.unit.surrogate.flow_mol==m.fs.unit.tube_inlet.flow_mol[0])
    # m.fs.unit.surrogate.constraint_p_inlet = Constraint(expr=m.fs.unit.surrogate.pressure==m.fs.unit.tube_inlet.pressure[0])

    # iscale.set_scaling_factor(m.fs.unit.surrogate.face_area, 1e2)
    # iscale.set_scaling_factor(m.fs.unit.surrogate.d_tube_outer, 1e2)
    # iscale.set_scaling_factor(m.fs.unit.surrogate.flow_mol, 1e0)
    # iscale.set_scaling_factor(m.fs.unit.surrogate.pressure, 1e-7)
    # iscale.calculate_scaling_factors(m.fs.unit.surrogate)

    # Loop for initializing temperature_wall based on the initial concrete temperature wall_t_init
    for i in m.fs.unit.segments_set:
            m.fs.unit.surrogate.temperature_wall[m.fs.unit.temperature_wall_index.ordered_data()[i-1]] = value(m.fs.unit.surrogate.wall_t_init[i])

    # Get default solver for testing
    solver = get_solver()

    print("="*58)
    print(" "*10,"Initializing surrogates for Sample: ",s+1," "*10)
    print("="*58)

    res = solver.solve(m.fs.unit.surrogate, tee=True)
    for i in m.fs.unit.segments_set:
        print('Wall temperature for segment {0}:'.format(i), value(m.fs.unit.surrogate.temperature_wall[m.fs.unit.temperature_wall_index.ordered_data()[i-1]]))
    # m.fs.unit.temperature_wall.display()

    m.fs.unit.tube_inlet.enth_mol[0].\
        fix(iapws95.htpx(df_test.loc[s,('T_fluid_sim_end')][1]*pyunits.K, m.fs.unit.tube_inlet.pressure[0].value*pyunits.Pa))  # K
    m.fs.unit.tube_heat_transfer_coefficient.fix(95.47/1.31)

    # Surrogate model for first length index
    # m.fs.unit.temperature_wall[0, :].fix(1000)

    # Surrogate model for second length index

    assert degrees_of_freedom(m) == 0

    solver.options = {
                "tol": 1e-6,
                "max_iter": 500,
                "halt_on_ampl_error": "yes",
                "bound_push": 1e-10,
                "mu_init": 1e-6
            }

    print("="*58)
    print(" "*10,"Initializing unit model for Sample: ",s+1," "*10)
    print("="*58)
    m.fs.unit.initialize(outlvl=idaeslog.DEBUG, optarg=solver.options)

    print("="*38)
    print(" "*10,"Solving Sample: ",s+1," "*10)
    print("="*38)
    res = solver.solve(m, tee=True)
    solution_state[s] = res.solver.termination_condition
    # m.fs.unit.temperature_wall.display()

    for i in m.fs.unit.temperature_wall_index:
        print('Segment {0}: {1}'.format(round(i[1]*19+1), value(m.fs.unit.tube.properties[i].temperature)))
        df_tfluid.loc[s,round(i[1]*19+1)] = value(m.fs.unit.tube.properties[i].temperature)

# df_tfluid.to_excel('tfluid_temp.xlsx')
print(solution_state)

for i in m.fs.unit.temperature_wall_index:
    print('Segment {0}: {1}'.format(round(i[1]*19+1), value(m.fs.unit.tube.properties[i].temperature)))
    df_tfluid.loc[s,round(i[1]*19+1)] = value(m.fs.unit.tube.properties[i].temperature)

# print(df_tfluid['30'])

for i in m.fs.unit.temperature_wall_index:
    print(value(m.fs.unit.tube.properties[i].temperature))

for i in m.fs.unit.temperature_wall_index:
    print(value(m.fs.unit.tube.properties[i].vapor_frac))

for i in m.fs.unit.temperature_wall_index:
    print(value(m.fs.unit.tube.properties[i].enth_mol))