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


def add_concrete_tes(m, data=None):

    # TODO:
    if  hasattr(m, 'Flowsheetblock'):  # If-clause has to be verified
        raise Exception('Block does not contain an IDAES flowsheet block: Flowsheetblock')

    elif data is None:
        raise Exception('add_concrete_tes() requires a dictionary with the following data: deltaP, T_concrete_init, tube_length, \
                             tube_od, face_area, heat_transfer_coefficient')

    else:

        # m = ConcreteModel()
        # m.fs = FlowsheetBlock(default={"dynamic": False})

        # m.fs.properties = iapws95.Iapws95ParameterBlock(default={
        #     "phase_presentation": iapws95.PhaseType.LG})

        m.tes = ConcreteTubeSide(
            default={"property_package": m.prop_water,
                    "flow_type": HeatExchangerFlowPattern.cocurrent,
                    "transformation_method": "dae.finite_difference",
                    "transformation_scheme": "BACKWARD",
                    "has_pressure_change": True,
                    "finite_elements": 19})

        add_surrogates(m.tes)

        for i in m.tes.temperature_wall_index:
            eq_wall = Constraint(
                expr=m.tes.temperature_wall[0, i[1]] ==
                m.tes.surrogate.temperature_wall[0, i[1]])
            setattr(m.tes, 'eq_wall{}'.format(i[1]), eq_wall)

        for i in m.tes.temperature_wall_index:
            m.tes.tube.deltaP[i].fix(data['deltaP'])

        # m.fs.tes.temperature_wall.setlb(300)
        # m.fs.tes.temperature_wall.setub(900)
        # for i in m.fs.tes.segments_set:
        #     m.fs.tes.surrogate.wall_t_init[i].fix(tinit_concrete[i-1])
        # m.fs.tes.surrogate.wall_t_init.display()

        # Setting initial concrete wall temperature
        for i in m.tes.segments_set:
            m.tes.surrogate.wall_t_init[i].fix(round(data['T_concrete_init'][i-1],4))
        # m.fs.tes.surrogate.wall_t_init.display()

        m.tes.tube_length.fix(data['tube_length'])
        m.tes.d_tube_outer.fix(data['tube_od'])
        m.tes.d_tube_inner.fix(m.tes.d_tube_outer.value)
        m.tes.tube_heat_transfer_coefficient.fix(data['heat_transfer_coefficient'])

        # Variable created by the add_surrogate method that mirrors the variables from the 1D side tube model
        m.tes.surrogate.tube_length.fix(m.tes.tube_length.value)
        m.tes.surrogate.face_area.fix(data['face_area'])
        m.tes.surrogate.d_tube_outer.fix(m.tes.d_tube_outer.value)

        # m.fs.tes.surrogate.constraint_tube_length = Constraint(expr=m.fs.tes.surrogate.tube_length==m.fs.tes.tube_length)
        # m.fs.tes.surrogate.constraint_face_area = Constraint(expr=m.fs.tes.surrogate.face_area==df_test.loc[41,('face_area')][0])
        # m.fs.tes.surrogate.constraint_tube_od = Constraint(expr=m.fs.tes.surrogate.d_tube_outer==m.fs.tes.d_tube_outer)
        # m.fs.tes.surrogate.constraint_mdot = Constraint(expr=m.fs.tes.surrogate.flow_mol==m.fs.tes.tube_inlet.flow_mol[0])
        # m.fs.tes.surrogate.constraint_p_inlet = Constraint(expr=m.fs.tes.surrogate.pressure==m.fs.tes.tube_inlet.pressure[0])

        # iscale.set_scaling_factor(m.fs.tes.surrogate.face_area, 1e2)
        # iscale.set_scaling_factor(m.fs.tes.surrogate.d_tube_outer, 1e2)
        # iscale.set_scaling_factor(m.fs.tes.surrogate.flow_mol, 1e0)
        # iscale.set_scaling_factor(m.fs.tes.surrogate.pressure, 1e-7)
        # iscale.calculate_scaling_factors(m.fs.tes.surrogate)

def initialize_tes(m, init_data=None):

    # TODO:
    if hasattr(m,'tes'):  # If-clause has to be revised
        raise Exception('A concrete TES unit has not been defined. Make sure to pass in a tes block as input to the method')

    elif init_data is None:
        raise Exception('initialize_tes() requires a dictionary with the following data: mdot, P_inlet, T_fluid_inlet. mdot should be the flow rate per tube')

    else:
        m.tube_inlet.flow_mol[0].fix(init_data['mdot'])  # mol/s
        m.tube_inlet.pressure[0].fix(init_data['P_inlet'])  # Pa
        m.surrogate.flow_mol.fix(m.tube_inlet.flow_mol[0].value)
        m.surrogate.pressure.fix(m.tube_inlet.pressure[0].value)

        # Loop for initializing temperature_wall based on the initial concrete temperature wall_t_init
        for i in m.segments_set:
                m.surrogate.temperature_wall[m.temperature_wall_index.ordered_data()[i-1]] = value(m.surrogate.wall_t_init[i])

        # Get default solver for testing
        solver = get_solver()

        # solver.options = {
        # "tol": 1e-6,
        # "max_iter": 500,
        # "halt_on_ampl_error": "yes",
        # "bound_push": 1e-10,
        # "mu_init": 1e-6
        # }

        print("="*58)
        print(" "*10,"Initializing Surrogates"," "*10)
        print("="*58,'\n')

        res = solver.solve(m.surrogate, tee=True)
    
        for i in m.segments_set:
            print('Wall temperature for segment {0}:'.format(i), value(m.surrogate.temperature_wall[m.temperature_wall_index.ordered_data()[i-1]]))
        # m.fs.unit.temperature_wall.display()

        m.tube_inlet.enth_mol[0].\
            fix(iapws95.htpx(init_data['T_fluid_inlet']*pyunits.K, m.tube_inlet.pressure[0].value*pyunits.Pa))  # K

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
        
        print('\n', "="*58)
        print(" "*10,"Initializing 1D side tube model"," "*10)
        print("="*58, '\n')
        m.initialize(outlvl=idaeslog.DEBUG, optarg=solver.options)

        print('\n', "="*58)
        print(" "*10,"Initializing TES unit model"," "*10)
        print("="*58, '\n')
        res = solver.solve(m, tee=True)
        # solution_state[s] = res.solver.termination_condition
        # m.fs.unit.temperature_wall.display()

        print("="*58)
        print(" "*10, "Printing steam temperature", " "*10)
        print("="*58, '\n')
        for i in m.temperature_wall_index:
            print('Segment {0}: {1}'.format(round(i[1]*19+1), value(m.tube.properties[i].temperature)))

        print("="*58)
        print(" "*10, "Printing vapor fraction", " "*10)
        print("="*58, '\n')
        for i in m.temperature_wall_index:
            print(value(m.tube.properties[i].vapor_frac))

        print("="*58)
        print(" "*10, "Printing concrete wall temperature", " "*10)
        print("="*58, '\n')
        for i in m.temperature_wall_index:
            print(value(m.temperature_wall[i]))

        # Unfixing state variables
        m.tube_inlet.flow_mol[0].unfix()  # mol/s
        m.tube_inlet.pressure[0].unfix()  # Pa
        m.surrogate.flow_mol.unfix()
        m.surrogate.pressure.unfix()
        m.tube_inlet.enth_mol[0].unfix()

        # TODO: Add constraints
        # m.tube_inlet.flow_mol[0] == m.surrogate.flow_mol
        # m.tube_inlet.pressure[0] == m.surrogate.pressure
