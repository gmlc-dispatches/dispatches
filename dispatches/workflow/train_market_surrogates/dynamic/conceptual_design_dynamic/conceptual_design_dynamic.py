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

#the rankine cycle is a directory above this one, so modify path
from pyomo.common.fileutils import this_file_dir
import sys, os, json
# sys.path.append(os.path.join(this_file_dir(),"../../../models/simple_case"))

# use renewable energy codes in 'RE_flowsheet.py'
# import specified functions instead of using *
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import add_wind, add_battery, \
    create_model, 

from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals
from pyomo.network import Arc
from pyomo.util.infeasible import log_close_to_bounds

# Import IDAES components
from idaes.core import FlowsheetBlock, UnitModelBlockData

# Import heat exchanger unit model
from idaes.generic_models.unit_models import Heater, PressureChanger
from idaes.generic_models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.power_generation.costing.power_plant_costing import get_PP_costing

# Import steam property package
from idaes.generic_models.properties.iapws95 import htpx, Iapws95ParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.util import get_solver
import idaes.logger as idaeslog
import pyomo.environ as pyo

from read_scikit_to_omlt import load_scikit_mlp
import json
import pickle

#omlt can encode the neural networks in Pyomo
import omlt
from omlt.neuralnet import NetworkDefinition

# import codes from Darice
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel
from dispatches.models.renewables_case.RE_flowsheet import *

# from Darice's codes import functions to build the multi period model
from wind_battery_LMP.py import wind_battery_variable_pairs, wind_battery_periodic_variable_pairs,
                                wind_battery_om_costs, initialize_mp, wind_battery_model, wind_battery_mp_block

# path for folder that has surrogate models
surrogate_dir = os.path.join(this_file_dir(),"../NN_model_params")

# load scaling and bounds for NN surrogates (rev and # of startups)

with open(os.path.join(surrogate_dir,"training_parameters_revenue.json"), 'rb') as f:
    rev_data = json.load(f)

with open(os.path.join(surrogate_dir,"training_parameters_nstartups.json"), 'rb') as f:
    nstartups_data = json.load(f)

with open(os.path.join(surrogate_dir,"training_parameters_ws.json"), 'rb') as f:
    dispatch_data = json.load(f)

# load scikit neural networks
with open(os.path.join(surrogate_dir,'scikit_revenue.pkl'), 'rb') as f:
    nn_rev = pickle.load(f)

with open(os.path.join(surrogate_dir,'scikit_nstartups.pkl'), 'rb') as f:
    nn_nstartups = pickle.load(f)

# load dispath frequency surrogate models
with open(os.path.join(surrogate_dir,'scikit_NN_ws.pkl'), 'rb') as f:
    nn_dispatch = pickle.load(f)

''' 
read the cluster centers (dispatch representative days)

check with Alex if we need these.
According to the model, the operation cost is a fixed value of system_capacity(pmax),
not related with the actual power generated (system capacity*capacity factor).
'''

# file_name = 'result_6400years_shuffled_30clusters_OD.json'
# with open(file_name, 'r') as f:
#     cluster_results = json.load(f)
# cluster_centers = np.array(cluster_results['model_params']['cluster_centers_'])
# cluster_centers = cluster_centers.reshape(30,24)


# load scikit models and create OMLT NetworkDefinition objects
#Revenue model definition
input_bounds_rev = list(zip(rev_data['xmin'],rev_data['xmax']))
scaling_object_rev = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                                            factor_inputs=rev_data['xstd_inputs'],
                                            offset_outputs=[rev_data['zm_revenue']],
                                            factor_outputs=[rev_data['zstd_revenue']])
net_rev_defn = load_scikit_mlp(nn_rev,scaling_object_rev,input_bounds_rev)


# Nstartup model definition
input_bounds_nstartups = list(zip(nstartups_data['xmin'],nstartups_data['xmax']))
scaling_object_nstartups = omlt.OffsetScaling(offset_inputs=nstartups_data['xm_inputs'],
                                              factor_inputs=nstartups_data['xstd_inputs'],
                                              offset_outputs=[nstartups_data['zm_nstartups']],
                                              factor_outputs=[nstartups_data['zstd_nstartups']])
net_nstartups_defn = load_scikit_mlp(nn_nstartups,scaling_object_nstartups,input_bounds_nstartups)


# the dispatch frequency surrogate
input_bounds_dispatch = list(zip(dispatch_data['xmin'],dispatch_data['xmax']))
scaling_object_dispatch = omlt.OffsetScaling(offset_inputs=dispatch_data['xm_inputs'],
                                             factor_inputs=dispatch_data['xstd_inputs'],
                                             offset_outputs=[dispatch_data['ws_mean']],
                                             factor_outputs=[dispatch_data['ws_std']])
net_dispatch_defn = load_scikit_mlp(nn_dispatch,scaling_object_dispatch,input_bounds_dispatch)

def conceptual_design_dynamic_RE(input_params, num_rep_days, wind_resource_config, verbose = False, plant_type = 'RE'):

    # create a wind+battery model
    
    if plant_type not in ['RE', 'NU', 'FOSSIL']:
        except TypeError:
            print('Wrong plant type')

    m = ConcreteModel()
    # m.scenario = Block()

    # add surrogate input to the model
    m.pmax = Var(within=NonNegativeReals, bounds=(dispatch_data['xmin'],dispatch_data['xmax']), initialize=dispatch_data['xmin'])
    m.pmin_multi = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)
    m.ramp_multi = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1.0,16.0), initialize=4.0)
    m.min_dn_multi = Var(within=NonNegativeReals, bounds=(0.5,2.0), initialize=4.0)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=5)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.startup_cst = Var(within=NonNegativeReals, bounds=(0,136), initialize=1)

    m.inputs = [m.pmax,m.pmin_multi,m.ramp_multi,m.min_up_time,m.min_dn_multi,m.marg_cst,m.no_load_cst,m.startup_cst]
    
    # add NN surrogates to the model using omlt
    ##############################
    # revenue surrogate
    ##############################
    m.rev_surrogate = Var()
    m.nn_rev = omlt.OmltBlock()
    formulation_rev = omlt.neuralnet.ReducedSpaceContinuousFormulation(net_rev_defn)
    m.nn_rev.build_formulation(formulation_rev,input_vars=m.inputs, output_vars=[m.rev_surrogate])
    # make rev non-negative
    m.rev = Expression(expr=0.5*pyo.sqrt(m.ws_surrogate**2 + 0.001**2) + 0.5*m.ws_surrogate)

    ##############################
    #nstartups surrogate
    ##############################
    m.nstartups_surrogate = Var()
    m.nn_nstartups = omlt.OmltBlock()

    # need a function to check the type of the plant. 
    # For renewable plant, startup cost is 0. 

    if plant_type is 'RE':
        m.nstartups.fix(0)
    else:
        formulation_nstartups = omlt.neuralnet.ReducedSpaceContinuousFormulation(net_nstartups_defn)
        m.nn_nstartups.build_formulation(formulation_nstartups,input_vars=m.inputs, output_vars=[m.nstartups_surrogate])
        m.nstartups = Expression(expr=0.5*pyo.sqrt(m.nstartups_surrogate**2 + 0.001**2) + 0.5*m.nstartups_surrogate)


    #build the formulation on the omlt blocks
    ##############################
    # dispatch frequency surrogate
    ##############################
    m.nn_dispatch = omlt.OmltBlock()
    formulation_dispatch = omlt.neuralnet.ReducedSpaceContinuousFormulation(net_dispatch_defn)
    m.dispatch_set = pyo.Set(initialize=range(0,num_rep_days))
    m.dispatch_surrogate = pyo.Var(m.dispatch_set,within=NonNegativeReals)
    m.nn_dispatch.build_formulation(formulation_dispatch,input_vars=m.inputs, output_vars=list(m.dispatch_surrogate.values()))

    # dispatch frequency flowsheet, each instance a model. 
    
    scenarios = []
    for i in range(num_rep_days):
        print('Creating instance', i)
        
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
        submodel = scenario.pyomo_model
        blks = scenario.get_active_process_blocks()
        # fix the initial soc and energy throughput
        blks[0].fs.battery.initial_state_of_charge.fix(0)
        blks[0].fs.battery.initial_energy_throughput.fix(0)

        # what these for?
        if input_params['design_opt']:
            for blk in blks:
                if not input_params['extant_wind']:
                    blk.fs.windpower.system_capacity.unfix()
                blk.fs.battery.nameplate_power.unfix()
        

        # submodel.wind_system_capacity = Var(domain=NonNegativeReals, 
        #     initialize=input_params['wind_mw'] * 1e3, 
        #     units=pyunits.kW, 
        #     bounds=(0, input_params['wind_mw_ub'] * 1e3)
        #     )

        scenario.wind_system_capacity = Expression(m.pmax * 1e3)
        scenario.battery_system_capacity = Var(
            domain=NonNegativeReals, 
            initialize=input_params['batt_mw'] * 1e3, 
            units=pyunits.kW
            )
        
        if input_params['design_opt']:
            for blk in blks:
                if not input_params['extant_wind']:
                    blk.fs.windpower.system_capacity.unfix()
                blk.fs.battery.nameplate_power.unfix()

        scenario.wind_max_p = Constraint(mp_wind_battery.pyomo_model.TIME, 
            rule=lambda b, t: blks[t].fs.windpower.system_capacity <= scenario.wind_system_capacity
            )
        scenario.battery_max_p = Constraint(mp_wind_battery.pyomo_model.TIME, 
            rule=lambda b, t: blks[t].fs.battery.nameplate_power <= scenario.battery_system_capacity
            )
        
        scenario.wind_cap_cost = pyo.Param(default=1550, mutable=True)
        if input_params['extant_wind']:
            scenario.wind_cap_cost.set_value(0.0)
        scenario.batt_cap_cost = pyo.Param(default=1200, mutable=True)
        
        scenario.total_cost = Expression(expr=scenario.wind_cap_cost + scenario.dispatch_frequency*sum([blk.wind_op_total_cost for blk in blks]))

        setattr(m, 'scenario_{}'.format(i), scenario)

        scenarios.append(scenario)

    # total cost for operation and captial
    m.plant_total_cost = Expression(expr = sum(scenarios[i].total_cost for i in range(num_rep_days)))
    
    # set objective functions
    m.obj = Objective(expr = m.plant_total_cost + m.nstartups - m.rev)
    
    # solve the model
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=verbose)

    return m

