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
import matplotlib.pyplot as plt


# use renewable energy codes in 'RE_flowsheet.py'
# import specified functions instead of using *
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

# sys.path.append(os.path.join(this_file_dir(),"../../../../../"))
from dispatches.case_studies.renewables_case.RE_flowsheet import add_wind, add_battery, \
    create_model 

from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, units as pyunits, RangeSet
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
from pathlib import Path

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
from dispatches.case_studies.renewables_case.wind_battery_LMP import wind_battery_variable_pairs, \
                                wind_battery_periodic_variable_pairs, wind_battery_om_costs, \
                                initialize_mp, wind_battery_model, wind_battery_mp_block
from dispatches.case_studies.renewables_case.double_loop_utils import read_rts_gmlc_wind_inputs
from dispatches_sample_data import rts_gmlc
wind_forecast_df = read_rts_gmlc_wind_inputs(rts_gmlc.source_data_path, "303_WIND_1")

# path for folder that has surrogate models
re_nn_dir = Path("/Users/dguittet/Projects/Dispatches/NN_models")

# load scaling and bounds for NN surrogates (rev and # of startups)

with open(re_nn_dir / "revenue" / "automation_RE_revenue.json", 'rb') as f:
    rev_data = json.load(f)

with open(re_nn_dir / "dispatch" / "RE_H2_dispatch_surrogate_param.json", 'rb') as f:
    dispatch_data = json.load(f)

# load keras neural networks

nn_rev = keras.models.load_model(re_nn_dir / "dispatch" / "RE_H2_dispatch_surrogate_model")

nn_dispatch = keras.models.load_model(re_nn_dir / "revenue" / "automation_RE_revenue")


# read the cluster centers (dispatch representative days)

with open(re_nn_dir / "RE_H2_20clusters.json", 'r') as f:
    cluster_results = json.load(f)
cluster_center = np.array(cluster_results['model_params']['cluster_centers_'])
cluster_center = cluster_center.reshape(-1, 24)

# add zero/full capacity days to the clustering results. 
full_days = np.array([np.ones(24)])
zero_days = np.array([np.zeros(24)])

# corresponds to the ws, ws[0] is for zero cf days and ws[31] is for full cd days.
cluster_centers = np.concatenate((zero_days, cluster_center, full_days), axis = 0)


# load keras models and create OMLT NetworkDefinition objects
#Revenue model definition
input_bounds_rev = {i:(rev_data['xmin'][i],rev_data['xmax'][i]) for i in range(len(rev_data['xmin']))}
scaling_object_rev = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                                            factor_inputs=rev_data['xstd_inputs'],
                                            offset_outputs=[rev_data['y_mean']],
                                            factor_outputs=[rev_data['y_std']])
net_rev_defn = load_keras_sequential(nn_rev,scaling_object_rev,input_bounds_rev)


# the dispatch frequency surrogate
input_bounds_dispatch = {i:(dispatch_data['xmin'][i],dispatch_data['xmax'][i]) for i in range(len(dispatch_data['xmin']))}
scaling_object_dispatch = omlt.OffsetScaling(offset_inputs=dispatch_data['xm_inputs'],
                                             factor_inputs=dispatch_data['xstd_inputs'],
                                             offset_outputs=dispatch_data['ws_mean'],
                                             factor_outputs=dispatch_data['ws_std'])
net_dispatch_defn = load_keras_sequential(nn_dispatch,scaling_object_dispatch,input_bounds_dispatch)


def conceptual_design_dynamic_RE(input_params, num_rep_days, verbose = False, plant_type = 'RE'):

    # create a wind+battery model
    
    if plant_type not in ['RE', 'NU', 'FOSSIL']:
        raise TypeError('Wrong plant type')
        

    m = ConcreteModel(name = 'RE_Conceptual_Design_full_surrogates')

    # add surrogate input to the model
    m.pmax = Var(within=NonNegativeReals, bounds=(177.5,443.75), initialize=177.5)      # Maximum Designed Capacity, unit = MW
    m.pmin_multi = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)     # Minimum Operating Multiplier
    m.ramp_multi = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)       # Ramp Rate Multiplier
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1.0,16.0), initialize=4.0)     # Minimum Up Time, unit = hr
    m.min_dn_multi = Var(within=NonNegativeReals, bounds=(0.5,2.0), initialize=2.0)     # Minimum Down Multiplier
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=5)             # Marginal Cost, unit = $/MWh
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)         # No load cost, unit = $/hr
    m.startup_cst = Var(within=NonNegativeReals, bounds=(0,136), initialize=1)          # Representative Startup Cost, unit = $/MW

    inputs = [m.pmax,m.pmin_multi,m.ramp_multi,m.min_up_time,m.min_dn_multi,m.marg_cst,m.no_load_cst,m.startup_cst]
    
    # add battery_system_capacity as a variable, unit = kW
    # add wind_system_capacity as a variable, unit = kW
    m.battery_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['batt_mw']*1000)
    m.wind_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['wind_mw']*1000)
    # m.wind_system_capacity = Var(domain=NonNegativeReals, initialize=input_params['wind_mw']*1000, bounds = (177.5*1000,input_params['wind_mw_ub']*1000))


    m.power_constraint = Constraint(expr = m.wind_system_capacity + m.battery_system_capacity >= m.pmax*1000)

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


    # make rev non-negative, MM$
    m.rev = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

    ##############################
    # dispatch frequency surrogate
    ##############################
    m.dis_set = RangeSet(0,num_rep_days-1)
    m.dispatch_surrogate = Var(m.dis_set, initialize = [(x+1)/(x+1)/num_rep_days for x in range(num_rep_days)])
    m.nn_dispatch = omlt.OmltBlock()
    formulation_dispatch = FullSpaceNNFormulation(net_dispatch_defn)
    m.nn_dispatch.build_formulation(formulation_dispatch)

    m.constraint_list_dispatch = ConstraintList()
    for i in range(len(inputs)):
        m.constraint_list_dispatch.add(inputs[i] == m.nn_dispatch.inputs[i])

    m.dispatch_surrogate_nonneg = Var(m.dis_set, initialize = [(x+1)/(x+1)/num_rep_days for x in range(num_rep_days)])
    for i in range(num_rep_days):
        # sum of outputs of dispatch surrogate may not be 1, scale them. 
        m.constraint_list_dispatch.add(m.dispatch_surrogate_nonneg[i] == 0.5*pyo.sqrt(m.nn_dispatch.outputs[i]**2 + 0.001**2) + 0.5*m.nn_dispatch.outputs[i])

    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(m.dispatch_surrogate[i] == m.dispatch_surrogate_nonneg[i]/sum(value(m.dispatch_surrogate_nonneg[j]) for j in range(num_rep_days)))

    # dispatch frequency flowsheet, each scenario is a model. 
    scenario_models = []
    for i in range(num_rep_days):
        print('Creating instance', i)

        # set the capacity factor from the clustering results.

        # updata on Aug 2 meeting: use default wind source data. 
        # at the output to grid, add a constraint (blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0]) = P_max*CF_i
        # For each scenario, use the same wind_speed profile at this moment. 
        clustered_capacity_factors = cluster_centers[i]
        input_params['wind_resource'] =  {t:
                                             {'wind_resource_config': {
                                                  'resource_speed': [wind_speeds[t]]}
                                             } for t in range(24)}
        
        scenario = MultiPeriodModel(
            n_time_points=24,
            process_model_func=partial(wind_battery_mp_block, input_params=input_params, verbose=verbose),
            linking_variable_func=wind_battery_variable_pairs,
            periodic_variable_func=wind_battery_periodic_variable_pairs,
            )
        
        # the dispatch frequency is determinated by the surrogate model
        
        # use our data to fix the dispatch power in the scenario

        scenario.build_multi_period_model(input_params['wind_resource'])
        scenario_model = scenario.pyomo_model
        blks = scenario.get_active_process_blocks()
        # fix the initial soc and energy throughput
        # blks[0].fs.battery.initial_state_of_charge.fix(0)
        # blks[0].fs.battery.initial_energy_throughput.fix(0)


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

        # scenario_model.wind_system_capacity = Expression(expr = m.wind_system_capacity)
        # scenario_model.battery_system_capacity = Expression(expr = m.battery_system_capacity)


        scenario_model.wind_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.windpower.system_capacity <= m.wind_system_capacity
            )
        scenario_model.battery_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.battery.nameplate_power <= m.battery_system_capacity
            )
        
        for t, blk in enumerate(blks):
            blk_wind = blk.fs.windpower
            blk_battery = blk.fs.battery
            
            # add operating costs for $/hr
            blk.operate_total_cost = Expression(
                expr=blk_wind.system_capacity * blk_wind.op_cost / 8760 + blk_battery.var_cost
            )
        
            # add a constraint at the output side to the grid
            # blk.output_const = Constraint(expr = blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0] >= m.pmax*1000*clustered_capacity_factors[t])
            blk.output_const = Constraint(expr = blk.fs.splitter.grid_elec[0] + blk_battery.elec_out[0] >= m.pmax*1000*clustered_capacity_factors[t])

        scenario_model.dispatch_frequency = Expression(expr = m.dispatch_surrogate[i])

        scenario_model.total_cost = Expression(expr=scenario_model.dispatch_frequency*365*sum([blk.operate_total_cost for blk in blks]))

        setattr(m, 'scenario_model_{}'.format(i), scenario_model)

        scenario_models.append(scenario_model)

    # total cost for operation and captial

    # extant_wind means if it is a built plant. If true, the captial cost is 0.

    m.wind_cap_cost = Param(default=1550, mutable=True)

    if input_params['extant_wind']:
        m.wind_cap_cost.set_value(0.0)

    m.batt_cap_cost = Param(default=1200, mutable=True)

    m.plant_cap_cost = Expression(expr = m.wind_cap_cost*m.wind_system_capacity + m.batt_cap_cost*m.battery_system_capacity)

    m.plant_operation_cost = Expression(expr = sum(scenario_models[i].total_cost for i in range(num_rep_days)))

    # startup cost in $
    m.plant_startup_cost = Expression(expr = m.nstartups*m.startup_cst*m.pmax)

    m.plant_total_cost = Expression(expr = m.plant_cap_cost + m.plant_operation_cost + m.plant_startup_cost)

    # set objective functions in $
    m.obj = Objective(expr = m.plant_total_cost - m.rev*1000000)
    
    return m



def record_result(m, num_rep_days):
    wind_to_grid = []
    batt_to_grid = []
    wind_to_batt = []
    wind_gen = []
    soc = []

    for i in range(num_rep_days):
        scenario_model = getattr(m, 'scenario_model_{}'.format(i))
        # print(value(scenario_model.blocks[j].process.fs.splitter.grid_elec[0]))
        # print(value(scenario_model.blocks[j].process.fs.battery.elec_out[0]))

        _wind_gen = [value(scenario_model.blocks[j].process.fs.windpower.electricity[0]) for j in range(24)]
        wind_gen.append(_wind_gen)

        _wind_to_grid = [value(scenario_model.blocks[j].process.fs.splitter.grid_elec[0]) for j in range(24)]
        wind_to_grid.append(_wind_to_grid )

        _batt_to_grid = [value(scenario_model.blocks[j].process.fs.battery.elec_out[0]) for j in range(24)]
        batt_to_grid.append(_batt_to_grid)

        _wind_to_batt = [value(scenario_model.blocks[j].process.fs.battery.elec_in[0]) for j in range(24)]
        wind_to_batt.append(_wind_to_batt)

        _soc = [value(scenario_model.blocks[j].process.fs.battery.state_of_charge[0]) for j in range(24)]
        soc.append(_soc)

    print("wind farm capacity = {} MW".format(value(m.wind_system_capacity)/1000))
    print("battery capacity = {} MW".format(value(m.battery_system_capacity)/1000))
    print("p_max = {}MW".format(value(m.pmax)))
    print("Plant captial cost = ${}".format(value(m.plant_cap_cost)))
    print("Plant operation cost within 1 year = ${}".format(value(m.plant_operation_cost)))
    print("plant revenue in this year = ${}".format(value(m.rev)))

    print('----------')
    for i in range(num_rep_days):
        print(value(m.dispatch_surrogate[i]))
    print('----------')
    # for i in range(num_rep_days):
    #     print(value(m.nn_dispatch.outputs[i]))
    # print('----------')
    # for i in range(num_rep_days):
    #     print(value(m.dispatch_surrogate_nonneg[i]))

    print('----------')
    print(sum(value(m.dispatch_surrogate[j]) for j in m.dis_set))
    
    hours = [t for t in range(24)]
    
    print('-------------------------')
    print(value(m.pmax))
    print(value(m.pmin_multi))
    print(value(m.ramp_multi))
    print(value(m.min_up_time))
    print(value(m.min_dn_multi))
    print(value(m.marg_cst))
    print(value(m.no_load_cst))
    print(value(m.startup_cst))
    

    title_font = {'fontsize': 16,'fontweight': 'bold'}
    for day in range(num_rep_days):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,9))
        ax1.plot(hours, wind_gen[day], label = 'wind_gen')
        ax1.plot(hours, wind_to_grid[day], label = 'wind_to_grid')
        ax1.plot(hours, wind_to_batt[day], label = 'wind_to_batt')
        ax1.plot(hours, batt_to_grid[day], label = 'batt_to_grid')
        ax2.plot(hours, soc[day], label = 'soc')
        ax1.set_title('wind+battery day_{}'.format(day),fontdict = title_font)
        ax2.set_title('soc, day_{}'.format(day),fontdict = title_font)
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('Time/hr',fontsize=16,fontweight='bold')
        ax1.set_ylabel('Power/kW',fontsize=16,fontweight='bold')
        ax2.set_xlabel('Time/hr',fontsize=16,fontweight='bold')
        ax2.set_ylabel('SOC/kWh',fontsize=16,fontweight='bold')
        ax1.tick_params(axis='both', labelsize=15)
        ax2.tick_params(axis='both', labelsize=15)
        ax1.tick_params(direction="in",top=True, right=True)
        ax2.tick_params(direction="in",top=True, right=True)
        ax1.set_ylim(-10000,value(m.wind_system_capacity)+20000)
        ax2.set_ylim(-10000,value(m.battery_system_capacity)*4+20000)
        fig.savefig("Two_plots_day_{}_run_3".format(day),dpi = 300)

        fig1,ax = plt.subplots(figsize=(9,6))
        total_output = []
        for a,b in zip(wind_to_grid[day],batt_to_grid[day]):
            total_output.append(a+b)
        ax.plot(hours, total_output, label = 'supply output')
        ax.plot(hours, value(m.pmax)*1000*cluster_centers[day], '--', label = 'demand profile')
        ax.set_xlabel('Time/hr',fontsize=16,fontweight='bold')
        ax.set_ylabel('Power/kW',fontsize=16,fontweight='bold')
        ax.tick_params(axis='both', labelsize=15)
        ax.tick_params(direction="in",top=True, right=True)
        ax.set_ylim(-10000,value(m.wind_system_capacity)+20000)
        ax.legend()
        ax.set_title('demand and supply profile_day_{}'.format(day),fontdict = title_font)
        ax.annotate("ws = {}".format(value(m.dispatch_surrogate[day])),(0,250000))
        fig1.savefig('demand_supply_day_{}_run_3'.format(day),dpi =300)