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

# conceptual_design_problem_dynamic formation 2, only use timeseries clustering to cluster dispatch data
# NN is based on dispatch_shuffled_data_0.csv, 32 clusters including capacity factor 0/1 days.
# use omlt v1.0 in this file.

#the rankine cycle is a directory above this one, so modify path
import json
import time
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, units as pyunits, RangeSet
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import omlt
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation
from omlt.io import load_keras_sequential
import idaes.logger as idaeslog
from idaes.core.solvers.get_solver import get_solver
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

from dispatches.case_studies.renewables_case.load_parameters import pem_bar, pem_temp, wind_cap_cost, \
                                wind_op_cost, pem_op_cost, pem_var_cost, pem_cap_cost, PA, h2_mols_per_kg
from dispatches.case_studies.renewables_case.wind_battery_PEM_LMP import wind_battery_pem_variable_pairs, \
                                wind_battery_pem_periodic_variable_pairs, wind_battery_pem_om_costs, \
                                initialize_mp, wind_battery_pem_model, wind_battery_pem_mp_block


# path for folder that has surrogate models
re_nn_dir = Path("/Users/dguittet/Projects/Dispatches/NN_models")

# load scaling and bounds for NN surrogates (rev and # of startups)

with open(re_nn_dir / "Wind_PEM_rt_revenue" / "RE_revenue_params.json", 'rb') as f:
    rev_data = json.load(f)

with open(re_nn_dir / "Wind_PEM_rt_dispatch" / "RE_H2_dispatch_surrogate_param_20.json", 'rb') as f:
    dispatch_data = json.load(f)

# load keras neural networks
# Input variables are PEM bid price, PEM MW, Reserve Factor and Load Shed Price
nn_rev = keras.models.load_model(re_nn_dir / "Wind_PEM_rt_revenue" / "RE_revenue")
nn_dispatch = keras.models.load_model(re_nn_dir / "Wind_PEM_rt_dispatch" / "RE_H2_dispatch_surrogate_model_20")


# read the cluster centers (dispatch representative days)

with open(re_nn_dir / "Wind_PEM_rt_dispatch" / "RE_224years_20clusters_OD.json", 'r') as f:
    cluster_results = json.load(f)
cluster_center = np.array(cluster_results['model_params']['cluster_centers_'])
dispatch_clusters = cluster_center[:, 0]
dispatch_clusters = dispatch_clusters.reshape(-1, 24)
resource_clusters = cluster_center[:, 1]
resource_clusters = resource_clusters.reshape(-1, 24)

# add zero/full capacity days to the clustering results. 
full_days = np.array([np.ones(24)])
zero_days = np.array([np.zeros(24)])

# corresponds to the ws, ws[0] is for zero cf days and ws[19] is for full cf days.
# dispatch_clusters = np.concatenate((zero_days, dispatch_clusters, full_days), axis = 0)


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


def conceptual_design_dynamic_RE(input_params, num_rep_days, PEM_bid=None, PEM_MW=None, verbose=False):

    m = ConcreteModel(name = 'RE_Conceptual_Design_full_surrogates')

    # add surrogate input to the model
    m.wind_system_capacity = Var(domain=NonNegativeReals, bounds=(100 * 1e3, 1000 * 1e3), initialize=input_params['wind_mw'] * 1e3)
    
    m.pem_system_capacity = Var(domain=NonNegativeReals, bounds=(127.05 * 1e3, 423.5 * 1e3), initialize=input_params['pem_mw'] * 1e3, units=pyunits.kW)
    m.pem_bid = Var(within=NonNegativeReals, bounds=(15, 45), initialize=45)                    # Energy Bid $/MWh
    m.reserve_percent = Param(within=NonNegativeReals, initialize=15)   # Reserves Fraction on Grid
    m.shortfall_price = Param(within=NonNegativeReals, initialize=1000)     # Energy price during load shed

    inputs = [m.pem_bid, m.pem_system_capacity * 1e-3 / 847 * m.wind_system_capacity * 1e-3, m.reserve_percent, m.shortfall_price]

    # extant_wind means if it is a built plant. If true, the captial cost is 0.
    if input_params['extant_wind']:
        m.wind_system_capacity.fix()
    if PEM_bid is not None:
        m.pem_bid.fix(PEM_bid)
    if PEM_MW is not None:
        m.pem_system_capacity.fix(PEM_MW * 1e3)

    # add NN surrogates to the model using omlt
    ##############################
    # revenue surrogate
    ##############################
    m.rev_surrogate = Var()
    m.nn_rev = omlt.OmltBlock()
    formulation_rev = FullSpaceNNFormulation(net_rev_defn)
    m.nn_rev.build_formulation(formulation_rev)
    
    m.constraint_list_rev = ConstraintList()
    for i in range(len(inputs)):
        m.constraint_list_rev.add(inputs[i] == m.nn_rev.inputs[i])
    m.constraint_list_rev.add(m.rev_surrogate == m.nn_rev.outputs[0])

    # make rev non-negative, MM$
    # m.rev = Expression(expr=0.5 * pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5 * m.rev_surrogate)
    m.rev = Expression(expr=m.rev_surrogate)

    ##############################
    # dispatch frequency surrogate
    ##############################
    m.dis_set = RangeSet(0, num_rep_days - 1)
    m.dispatch_surrogate = Var(m.dis_set, initialize=[(x + 1) / (x + 1) / num_rep_days for x in range(num_rep_days)])
    m.nn_dispatch = omlt.OmltBlock()
    formulation_dispatch = FullSpaceNNFormulation(net_dispatch_defn)
    m.nn_dispatch.build_formulation(formulation_dispatch)

    m.constraint_list_dispatch = ConstraintList()
    for i in range(len(inputs)):
        m.constraint_list_dispatch.add(inputs[i] == m.nn_dispatch.inputs[i])

    m.dispatch_surrogate_nonneg = Var(m.dis_set, initialize=[(x + 1) / (x + 1) / num_rep_days for x in range(num_rep_days)])
    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(m.dispatch_surrogate_nonneg[i] == 0.5 * pyo.sqrt(m.nn_dispatch.outputs[i]**2 + 0.001**2) + 0.5 * m.nn_dispatch.outputs[i])
    
    # sum of outputs of dispatch surrogate may not be 1, scale them. 
    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(m.dispatch_surrogate[i] == m.dispatch_surrogate_nonneg[i] / sum(value(m.dispatch_surrogate_nonneg[j]) for j in range(num_rep_days)))

    # dispatch frequency flowsheet, each scenario is a model. 
    scenario_models = []
    for i in range(num_rep_days):
        # set the capacity factor from the clustering results.
        # For each scenario, use the same wind_speed profile at this moment. 
        clustered_capacity_factors = dispatch_clusters[i]
        clustered_wind_resource = resource_clusters[i]
        
        input_params['wind_resource'] = {t: {'wind_resource_config': {
                                                'capacity_factor': 
                                                    [clustered_wind_resource[t]]}} for t in range(24)}

        scenario = MultiPeriodModel(
            n_time_points=24,
            process_model_func=partial(wind_battery_pem_mp_block, input_params=input_params, verbose=verbose),
            linking_variable_func=wind_battery_pem_variable_pairs,
            periodic_variable_func=wind_battery_pem_periodic_variable_pairs,
            )
        
        scenario.build_multi_period_model(input_params['wind_resource'])
        scenario_model = scenario.pyomo_model
        blks = scenario.get_active_process_blocks()

        # unfix wind for design, PEM is not fixed, leave battery fixed at 0 since no battery
        for blk in blks:
            if not input_params['extant_wind']:
                blk.fs.windpower.system_capacity.unfix()

        '''
        Differ from the wind_battery_LMP.py, in our problem, the wind farm 
        capacity is a design variable which we want to optimize. So we have every scenario's 
        wind_system_capacity is equal to the pmax*1e3, the unit is kW.
        '''

        scenario_model.wind_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.windpower.system_capacity <= m.wind_system_capacity)
        scenario_model.pem_max_p = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.pem.electricity[0] <= m.pem_system_capacity)

        scenario_model.dispatch_frequency = Expression(expr=m.dispatch_surrogate[i])

        scenario_model.hydrogen_produced = Expression(scenario_model.TIME,
            rule=lambda b, t: blks[t].fs.pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600)
        scenario_model.hydrogen_revenue = Expression(
            expr=scenario_model.dispatch_frequency * 365 * sum(scenario_model.hydrogen_produced[t] for t in scenario_model.TIME) * input_params['h2_price_per_kg'])
        scenario_model.op_var_cost = Expression( 
            expr=sum(input_params['pem_var_cost'] * blks[t].fs.pem.electricity[0] for t in scenario_model.TIME))
        scenario_model.var_total_cost = Expression(expr=scenario_model.dispatch_frequency * 365 * scenario_model.op_var_cost)
        scenario_model.output_const = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.splitter.grid_elec[0] == m.wind_system_capacity * clustered_capacity_factors[t])

        setattr(m, 'scenario_model_{}'.format(i), scenario_model)
        scenario_models.append(scenario_model)

    m.plant_cap_cost = Expression(
        expr=input_params['wind_cap_cost'] * m.wind_system_capacity + input_params['pem_cap_cost'] * m.pem_system_capacity)
    m.annual_fixed_cost = pyo.Expression(
        expr=m.wind_system_capacity * input_params["wind_op_cost"] + m.pem_system_capacity * input_params["pem_op_cost"])
    m.plant_operation_cost = Expression(
        expr=sum(scenario_models[i].var_total_cost for i in range(num_rep_days)))
    m.hydrogen_rev = Expression(
        expr=sum(scenario_models[i].hydrogen_revenue for i in range(num_rep_days)))

    m.NPV = Expression(expr=-m.plant_cap_cost + PA * (m.rev + m.hydrogen_rev - m.plant_operation_cost - m.annual_fixed_cost))
    m.obj = Objective(expr=-m.NPV * 1e-8)
    
    return m


def record_result(m, num_rep_days, plotting=False):
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []

    for i in range(num_rep_days):
        scenario_model = getattr(m, 'scenario_model_{}'.format(i))

        _wind_gen = [value(scenario_model.blocks[j].process.fs.windpower.electricity[0]) for j in range(24)]
        wind_gen.append(_wind_gen)

        _wind_to_grid = [value(scenario_model.blocks[j].process.fs.splitter.grid_elec[0]) for j in range(24)]
        wind_to_grid.append(_wind_to_grid )

        _wind_to_pem = [value(scenario_model.blocks[j].process.fs.splitter.pem_elec[0]) for j in range(24)]
        wind_to_pem.append(_wind_to_pem)

    results = {
        "wind_mw": value(m.wind_system_capacity) * 1e-3,
        "pem_mw": value(m.pem_system_capacity) * 1e-3,
        "pem_bid": value(m.pem_bid),
        "e_revenue": value(m.rev),
        "h_revenue": value(m.hydrogen_rev),
        "NPV": value(m.NPV)
    }

    for day in range(num_rep_days):
        results[f'freq_day_{day}'] = value(m.dispatch_surrogate[day])

    print("Wind capacity = {} MW".format(value(m.wind_system_capacity) * 1e-3))
    print("PEM capacity = {}MW".format(value(m.pem_system_capacity) * 1e-3))
    print("Plant bid = ${}".format(value(m.pem_bid)))
    print("Plant Revenue Annual = ${}".format(value(m.rev)))
    print("Plant NPV = ${}".format(value(m.NPV)))

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
    
    if plotting:

        title_font = {'fontsize': 16,'fontweight': 'bold'}
        for day in range(num_rep_days):
            fig, axs = plt.subplots(1, 2, figsize=(16,9))
            axs[0].plot(hours, wind_gen[day], label = 'wind_gen')
            axs[0].plot(hours, wind_to_grid[day], label = 'wind_to_grid')
            axs[0].set_title('wind day_{}'.format(day),fontdict = title_font)
            axs[0].legend()
            axs[0].set_xlabel('Time/hr',fontsize=16,fontweight='bold')
            axs[0].set_ylabel('Power/kW',fontsize=16,fontweight='bold')
            axs[0].tick_params(axis='both', labelsize=15)
            axs[0].tick_params(direction="in",top=True, right=True)
            axs[0].set_ylim(0,value(m.wind_system_capacity) * 1.02)
            # fig.savefig("Two_plots_day_{}_run_3".format(day),dpi = 300)

            total_output = wind_to_grid[day]
            axs[1].plot(hours, wind_to_pem[day], label = 'wind_to_pem')
            # axs[1].plot(hours, value(m.wind_system_capacity)*1000*dispatch_clusters[day], '--', label = 'demand profile')
            axs[1].set_xlabel('Time/hr',fontsize=16,fontweight='bold')
            axs[1].set_ylabel('Power/kW',fontsize=16,fontweight='bold')
            axs[1].tick_params(axis='both', labelsize=15)
            axs[1].tick_params(direction="in",top=True, right=True)
            axs[1].set_ylim(0,value(m.pem_system_capacity) * 1.02)
            axs[1].legend()
            axs[1].set_title('wind to PEM day_{}'.format(day),fontdict = title_font)
            axs[1].annotate("ws = {}".format(value(m.dispatch_surrogate[day])),(0,250000))
            plt.show()
    return results


def run_design(PEM_bid=None, PEM_size=None):
    model = conceptual_design_dynamic_RE(default_input_params, num_rep_days=n_rep_days, PEM_bid=PEM_bid, PEM_MW=PEM_size, verbose=False)
    nlp_solver = SolverFactory('ipopt')
    # nlp_solver.options['max_iter'] = 500
    nlp_solver.options['acceptable_tol'] = 1e-8
    nlp_solver.solve(model, tee=True)
    return record_result(model, n_rep_days, plotting=True)

default_input_params = {
    "wind_mw": 847,
    "wind_mw_ub": 10000,
    "pem_mw": 423,
    "batt_mw": 0,
    "batt_mwh": 0,
    "pem_bar": pem_bar,
    "pem_temp": pem_temp,
    "tank_size": None,
    "tank_type": None,
    "turb_mw": None,

    "h2_price_per_kg": 3,
    "DA_LMPs": None,

    "design_opt": True,
    "extant_wind": True,        # fixed because parameter sweeps didn't change wind size

    "wind_cap_cost": wind_cap_cost,
    "wind_op_cost": wind_op_cost,
    "pem_cap_cost": pem_cap_cost,
    "pem_op_cost": pem_op_cost,
    "pem_var_cost": pem_var_cost
} 

start_time = time.time()
n_rep_days = dispatch_clusters.shape[0]


if __name__ == "__main__":
    result = run_design()
    exit()

    import multiprocessing as mp
    from itertools import product

    bids = np.linspace(15, 45, 13)
    sizes = np.linspace(127.05, 423.5, 15)
    inputs = product(bids, sizes)

    with mp.Pool(processes=4) as p:
        res = p.starmap(run_design, inputs)

    df = pd.DataFrame(res)
    df.to_csv("surrogate_results.csv")
