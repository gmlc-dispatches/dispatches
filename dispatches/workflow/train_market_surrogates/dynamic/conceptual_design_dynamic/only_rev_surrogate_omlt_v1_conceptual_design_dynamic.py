#################################################################################
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
#################################################################################

# conceptual_design_problem_dynamic formation 2, only use timeseries clustering to cluster dispatch data
# NN is based on dispatch_shuffled_data_0.csv, 32 clusters including capacity factor 0/1 days.
# use omlt v1.0 in this file.

#the rankine cycle is a directory above this one, so modify path
from pyomo.common.fileutils import this_file_dir
import sys, os, json
from functools import partial
import numpy as np

# use renewable energy codes in 'RE_flowsheet.py'
# import specified functions instead of using *
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

sys.path.append(os.path.join(this_file_dir(),"../../../../../"))
from dispatches.models.renewables_case.RE_flowsheet import add_wind, add_battery, \
    create_model 

from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, units as pyunits
from pyomo.network import Arc
from pyomo.util.infeasible import log_close_to_bounds

# Import IDAES components
from idaes.core import FlowsheetBlock, UnitModelBlockData

# Import heat exchanger unit model
from idaes.models.unit_models.heater import Heater
from idaes.models.unit_models.pressure_changer import PressureChanger
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models_extra.power_generation.costing.power_plant_costing import get_PP_costing

# Import steam property package
from idaes.models.properties.iapws95 import htpx, Iapws95ParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
import pyomo.environ as pyo

# from read_scikit_to_omlt import load_scikit_mlp
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#omlt can encode the neural networks in Pyomo
import omlt
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation
from omlt.io import load_keras_sequential

# import codes from Darice
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

# from Darice's codes import functions to build the multi period model
from dispatches.models.renewables_case.wind_battery_LMP import wind_battery_variable_pairs, \
                                wind_battery_periodic_variable_pairs, wind_battery_om_costs, \
                                initialize_mp, wind_battery_model, wind_battery_mp_block

# path for folder that has surrogate models
surrogate_dir = os.path.join(this_file_dir(),"../NN_model_params_keras")

# load scaling and bounds for NN surrogates (rev and # of startups)

with open(os.path.join(surrogate_dir,"keras_training_parameters_revenue.json"), 'rb') as f:
    rev_data = json.load(f)

with open(os.path.join(surrogate_dir,"keras_training_parameters_nstartups.json"), 'rb') as f:
    nstartups_data = json.load(f)

with open(os.path.join(surrogate_dir,"keras_training_parameters_ws_sigmoid.json"), 'rb') as f:
    dispatch_data = json.load(f)

# load keras neural networks

nn_rev = keras.models.load_model(os.path.join(surrogate_dir,"keras_revenue"))

nn_nstartups = keras.models.load_model(os.path.join(surrogate_dir,"keras_nstartups"))

nn_dispatch = keras.models.load_model(os.path.join(surrogate_dir,"keras_dispatch_frequency"))


# read the cluster centers (dispatch representative days)

file_name = 'result_6400years_shuffled_30clusters_OD.json'
with open(file_name, 'r') as f:
    cluster_results = json.load(f)
cluster_centers = np.array(cluster_results['model_params']['cluster_centers_'])
cluster_centers = cluster_centers.reshape(30,24)


# load keras models and create OMLT NetworkDefinition objects
#Revenue model definition
input_bounds_rev = {i:(rev_data['xmin'][i],rev_data['xmax'][i]) for i in range(len(rev_data['xmin']))}
scaling_object_rev = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                                            factor_inputs=rev_data['xstd_inputs'],
                                            offset_outputs=[rev_data['zm_revenue']],
                                            factor_outputs=[rev_data['zstd_revenue']])
net_rev_defn = load_keras_sequential(nn_rev,scaling_object_rev,input_bounds_rev)


# Nstartup model definition
input_bounds_nstartups = {i:(nstartups_data['xmin'][i],nstartups_data['xmax'][i]) for i in range(len(nstartups_data['xmin']))}
scaling_object_nstartups = omlt.OffsetScaling(offset_inputs=nstartups_data['xm_inputs'],
                                              factor_inputs=nstartups_data['xstd_inputs'],
                                              offset_outputs=[nstartups_data['zm_nstartups']],
                                              factor_outputs=[nstartups_data['zstd_nstartups']])
net_nstartups_defn = load_keras_sequential(nn_nstartups,scaling_object_nstartups,input_bounds_nstartups)


# the dispatch frequency surrogate
# input_bounds_dispatch = {i:(dispatch_data['xmin'][i],dispatch_data['xmax'][i]) for i in range(len(dispatch_data['xmin']))}
# scaling_object_dispatch = omlt.OffsetScaling(offset_inputs=dispatch_data['xm_inputs'],
#                                              factor_inputs=dispatch_data['xstd_inputs'],
#                                              offset_outputs=dispatch_data['ws_mean'],
#                                              factor_outputs=dispatch_data['ws_std'])
# net_dispatch_defn = load_keras_sequential(nn_dispatch,scaling_object_dispatch,input_bounds_dispatch)

def conceptual_design_dynamic_RE(input_params, num_rep_days, verbose = False, plant_type = 'RE'):

    # create a wind+battery model
    
    if plant_type not in ['RE', 'NU', 'FOSSIL']:
        raise TypeError('Wrong plant type')
        

    m = ConcreteModel(name = 'RE_Conceptual_Design_only_rev_NN')

    # add surrogate input to the model
    m.pmax = Var(within=NonNegativeReals, bounds=(dispatch_data['xmin'][0],dispatch_data['xmax'][0]), initialize=dispatch_data['xmin'][0])
    m.pmin_multi = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)
    m.ramp_multi = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1.0,16.0), initialize=4.0)
    m.min_dn_multi = Var(within=NonNegativeReals, bounds=(0.5,2.0), initialize=2.0)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=5)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.startup_cst = Var(within=NonNegativeReals, bounds=(0,136), initialize=1)

    inputs = [m.pmax,m.pmin_multi,m.ramp_multi,m.min_up_time,m.min_dn_multi,m.marg_cst,m.no_load_cst,m.startup_cst]
    
    # add NN surrogates to the model using omlt
    ##############################
    # revenue surrogate
    ##############################
    m.rev_surrogate = Var()
    m.nn_rev = omlt.OmltBlock()
    formulation_rev = FullSpaceNNFormulation(net_rev_defn)
    m.nn_rev.build_formulation(formulation_rev)
    
    m.constraint_list_rev = ConstraintList()

    for i in range(8):
        m.constraint_list_rev.add(inputs[i] == m.nn_rev.inputs[i])
    
    m.constraint_list_rev.add(m.rev_surrogate == m.nn_rev.outputs[0])


    # make rev non-negative
    m.rev = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

    ##############################
    #nstartups surrogate
    ##############################
    # m.nstartups_surrogate = Var()
    # m.nn_nstartups = omlt.OmltBlock()

    # need a function to check the type of the plant. 
    # For renewable plant, startup cost is 0.
    # For this only rev surrogate case, fix startup cost to 0. 

    if plant_type == 'RE':
        m.nstartups = Param(default=0)
    else:
        m.nstartups = Param(default=0)


    ##############################
    # dispatch frequency surrogate
    ##############################
    # For this only rev surrogate case, fix the dispatch frequency to some values.
    m.dis_set = Set(initialize = [0,1,2])
    m.dispatch_surrogate = Param(m.dis_set, initialize = [0.4,0.4,0.2])


    # dispatch frequency flowsheet, each scenario is a model. 
    
    scenario_models = []
    for i in range(num_rep_days):
        print('Creating instance', i)

        # set the capacity factor from the clustering results.
        clustered_capacity_factors = cluster_centers[i]
        input_params['wind_resource'] =  {t:
                                             {'wind_resource_config': {
                                                  'capacity_factor': [clustered_capacity_factors.tolist()[t]]}
                                             } for t in range(24)}
        
        scenario = MultiPeriodModel(
            n_time_points=24,
            process_model_func=partial(wind_battery_mp_block, input_params=input_params, verbose=verbose),
            linking_variable_func=wind_battery_variable_pairs,
            periodic_variable_func=wind_battery_periodic_variable_pairs,
            )
        
        # the dispatch frequency is determinated by the surrogate model
        scenario.dispatch_frequency = Expression(expr=0.5*pyo.sqrt(m.dispatch_surrogate[i]**2 + 0.001**2) + 0.5*m.dispatch_surrogate[i])
        
        # use our data to fix the dispatch power in the scenario

        scenario.build_multi_period_model(input_params['wind_resource'])
        scenario_model = scenario.pyomo_model
        blks = scenario.get_active_process_blocks()
        # fix the initial soc and energy throughput
        blks[0].fs.battery.initial_state_of_charge.fix(0)
        blks[0].fs.battery.initial_energy_throughput.fix(0)

        # what are these for?
        if input_params['design_opt']:
            for blk in blks:
                if not input_params['extant_wind']:
                    blk.fs.windpower.system_capacity.unfix()
                blk.fs.battery.nameplate_power.unfix()

        '''
        Differ from the wind_battery_LMP.py, in our problem, the wind farm 
        capacity is a design variable which we want to optimize. So we have every scenario's 
        wind_system_capacity is equal to the pmax*1e3, the unit is kW.
        '''

        scenario_model.wind_system_capacity = Expression(expr = m.pmax * 1e3)
        scenario_model.battery_system_capacity = Var(
            domain=NonNegativeReals, 
            initialize=input_params['batt_mw'] * 1e3, 
            units=pyunits.kW
            )


        scenario_model.wind_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.windpower.system_capacity <= scenario_model.wind_system_capacity
            )
        scenario_model.battery_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.battery.nameplate_power <= scenario_model.battery_system_capacity
            )
        
        for blk in blks:
            blk_wind = blk.fs.windpower
            blk_battery = blk.fs.battery
            
            # add operating costs
            blk.op_total_cost = Expression(
                expr=blk_wind.system_capacity * blk_wind.op_cost / 8760
            )



        scenario_model.wind_cap_cost = pyo.Param(default=1550, mutable=True)

        # extant_wind means if it is a built plant. If true, the captial cost is 0.
        if input_params['extant_wind']:
            scenario_model.wind_cap_cost.set_value(0.0)

        scenario_model.batt_cap_cost = pyo.Param(default=1200, mutable=True)
        
        scenario_model.dispatch_frequency = Expression(expr = m.dispatch_surrogate[i])

        scenario_model.total_cost = Expression(expr=scenario_model.wind_cap_cost + scenario_model.dispatch_frequency*sum([blk.op_total_cost for blk in blks]))

        setattr(m, 'scenario_model_{}'.format(i), scenario_model)

        scenario_models.append(scenario_model)

    # total cost for operation and captial
    m.plant_total_cost = Expression(expr = sum(scenario_models[i].total_cost for i in range(num_rep_days)))
    
    # set objective functions
    m.obj = Objective(expr = m.plant_total_cost + m.nstartups - m.rev)
    
    # solve the model
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=True)
    # m.pprint()
    return m

