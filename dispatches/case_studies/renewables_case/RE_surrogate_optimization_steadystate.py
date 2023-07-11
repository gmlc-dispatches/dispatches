#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
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
import json
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, units as pyunits, RangeSet
from pyomo.util.infeasible import log_infeasible_constraints, log_infeasible_bounds, log_close_to_bounds
import tensorflow as tf
from tensorflow import keras
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


# RT market only or Both RT and DA markets
rt_market_only = True
include_wind_capital_cost = False
shortfall = 1000

# path for folder that has surrogate models
re_nn_dir = Path(__file__).parent / "data" / "steady_state_surrogate"

def load_surrogate_model(re_nn_dir):
    # load scaling and bounds for NN surrogates (rev and # of startups)
    with open(re_nn_dir / "dispatch_frequency" / "static_clustering_wind_pmax.pkl", 'rb') as f:
        model = pickle.load(f)
    centers = model.cluster_centers_
    dispatch_clusters_mean = centers[:, 0]
    pem_clusters_mean = centers[:, 1]
    resource_clusters_mean = centers[:, 2]

    # load keras neural networks
    # Input variables are PEM bid price, PEM MW, Reserve Factor and Load Shed Price
    with open(re_nn_dir / "dispatch_frequency" / "ss_surrogate_param_wind_pmax.json", 'r') as f:
        dispatch_data = json.load(f)
    nn_dispatch = keras.models.load_model(re_nn_dir / "dispatch_frequency" / "ss_surrogate_model_wind_pmax")

    if rt_market_only:
        rev_data_f = re_nn_dir / "rt_revenue" / "RE_RT_revenue_params_2_25.json"
        nn_rev = keras.models.load_model(re_nn_dir / "rt_revenue" / "RE_RT_revenue_2_25")
    else:
        rev_data_f = re_nn_dir / "revenue" / "RE_revenue_params_2_25.json"
        nn_rev = keras.models.load_model(re_nn_dir / "revenue" / "RE_revenue_2_25")

    with open(rev_data_f, 'rb') as f:
        rev_data = json.load(f)

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
    net_frequency_defn = load_keras_sequential(nn_dispatch,scaling_object_dispatch,input_bounds_dispatch)
    return net_rev_defn, net_frequency_defn, dispatch_clusters_mean, pem_clusters_mean, resource_clusters_mean

def conceptual_design_dynamic_RE(input_params, PEM_bid=None, PEM_MW=None, verbose=False):

    net_rev_defn, net_frequency_defn, dispatch_clusters_mean, pem_clusters_mean, resource_clusters_mean = load_surrogate_model(re_nn_dir)

    num_rep_days = len(dispatch_clusters_mean) 

    m = ConcreteModel(name = 'RE_Conceptual_Design_full_surrogates')

    # add surrogate input to the model
    m.wind_system_capacity = Var(domain=NonNegativeReals, bounds=(100 * 1e3, 1000 * 1e3), initialize=input_params['wind_mw'] * 1e3)
    
    m.pem_system_capacity = Var(domain=NonNegativeReals, bounds=(127.05 * 1e3, 423.5 * 1e3), initialize=input_params['pem_mw'] * 1e3, units=pyunits.kW)
    m.pem_bid = Var(within=NonNegativeReals, bounds=(15, 45), initialize=45)                    # Energy Bid $/MWh
    m.reserve_percent = Param(within=NonNegativeReals, initialize=15)   # Reserves Fraction on Grid
    m.shortfall_price = Param(within=NonNegativeReals, initialize=shortfall)     # Energy price during load shed

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
    m.frequency_surrogate = Var(m.dis_set, initialize=[(x + 1) / (x + 1) / num_rep_days for x in range(num_rep_days)])
    m.nn_frequency = omlt.OmltBlock()
    formulation_frequency = FullSpaceNNFormulation(net_frequency_defn)
    m.nn_frequency.build_formulation(formulation_frequency)

    m.constraint_list_dispatch = ConstraintList()
    for i in range(len(inputs)):
        m.constraint_list_dispatch.add(inputs[i] == m.nn_frequency.inputs[i])

    m.frequency_surrogate_nonneg = Var(m.dis_set, initialize=[(x + 1) / (x + 1) / num_rep_days for x in range(num_rep_days)])
    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(m.frequency_surrogate_nonneg[i] == 0.5 * pyo.sqrt(m.nn_frequency.outputs[i]**2 + 0.001**2) + 0.5 * m.nn_frequency.outputs[i])
    
    # sum of outputs of dispatch surrogate may not be 1, scale them. 
    for i in range(num_rep_days):
        m.constraint_list_dispatch.add(m.frequency_surrogate[i] == m.frequency_surrogate_nonneg[i] / sum(value(m.frequency_surrogate_nonneg[j]) for j in range(num_rep_days)))

    # dispatch frequency flowsheet, each scenario is a model. 
    scenario_models = []
    for i in range(num_rep_days):
        # set the capacity factor from the clustering results.
        # For each scenario, use the same wind_speed profile at this moment. 
        dispatch_capacity_factor = dispatch_clusters_mean[i]
        pem_capacity_factor = pem_clusters_mean[i]
        wind_capacity_factor = resource_clusters_mean[i]
        
        input_params['wind_resource'] = {0: {'wind_resource_config': {
                                                'capacity_factor': 
                                                    wind_capacity_factor}}}
                                                    
        scenario = MultiPeriodModel(
            n_time_points=1,
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

        scenario_model.dispatch_frequency = Expression(expr=m.frequency_surrogate[i])
        scenario_model.output_const = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.splitter.grid_elec[0] == m.wind_system_capacity * dispatch_capacity_factor)
        scenario_model.pem_const = Constraint(scenario_model.TIME, 
            rule=lambda b, t: blks[t].fs.splitter.pem_elec[0] <= m.wind_system_capacity * pem_capacity_factor)

        scenario_model.elec_grid = Expression(
            expr=scenario_model.dispatch_frequency * 8760 * blks[0].fs.splitter.grid_elec[0])
        scenario_model.elec_pem = Expression(
            expr=scenario_model.dispatch_frequency * 8760 * blks[0].fs.splitter.pem_elec[0])
        scenario_model.hydrogen_produced = Expression(scenario_model.TIME,
            rule=lambda b, t: scenario_model.dispatch_frequency * 8760 * blks[t].fs.pem.outlet.flow_mol[0] / h2_mols_per_kg * 3600)
        scenario_model.hydrogen_revenue = Expression(
            expr=scenario_model.hydrogen_produced[0] * input_params['h2_price_per_kg'])
        scenario_model.op_var_cost = Expression( 
            expr=input_params['pem_var_cost'] * blks[0].fs.pem.electricity[0])
        scenario_model.var_total_cost = Expression(expr=scenario_model.dispatch_frequency * 8760 * scenario_model.op_var_cost)

        setattr(m, 'scenario_model_{}'.format(i), scenario_model)
        scenario_models.append(scenario_model)

    m.plant_cap_cost = Expression(
        expr=input_params['wind_cap_cost'] * m.wind_system_capacity * int(include_wind_capital_cost) + input_params['pem_cap_cost'] * m.pem_system_capacity)
    m.annual_fixed_cost = pyo.Expression(
        expr=m.wind_system_capacity * input_params["wind_op_cost"] + m.pem_system_capacity * input_params["pem_op_cost"])
    m.plant_operation_cost = Expression(
        expr=sum(scenario_models[i].var_total_cost for i in range(num_rep_days)))
    m.hydrogen_rev = Expression(
        expr=sum(scenario_models[i].hydrogen_revenue for i in range(num_rep_days)))

    m.NPV = Expression(expr=-m.plant_cap_cost + PA * (m.rev + m.hydrogen_rev - m.plant_operation_cost - m.annual_fixed_cost))
    m.NPV_ann = Expression(expr=-m.plant_cap_cost / PA + (m.rev + m.hydrogen_rev - m.plant_operation_cost - m.annual_fixed_cost))
    m.obj = Objective(expr=-m.NPV * 1e-8)
    
    return m, num_rep_days


def record_result(m, num_rep_days):
    wind_to_grid = []
    wind_to_pem = []
    wind_gen = []

    for i in range(num_rep_days):
        scenario_model = getattr(m, 'scenario_model_{}'.format(i))

        _wind_gen = [value(scenario_model.blocks[j].process.fs.windpower.electricity[0]) for j in range(1)]
        wind_gen.append(_wind_gen)

        _wind_to_grid = [value(scenario_model.blocks[j].process.fs.splitter.grid_elec[0]) for j in range(1)]
        wind_to_grid.append(_wind_to_grid )

        _wind_to_pem = [value(scenario_model.blocks[j].process.fs.splitter.pem_elec[0]) for j in range(1)]
        wind_to_pem.append(_wind_to_pem)

    results = {
        "wind_mw": value(m.wind_system_capacity) * 1e-3,
        "pem_mw": value(m.pem_system_capacity) * 1e-3,
        "pem_bid": value(m.pem_bid),
        "e_revenue": value(m.rev),
        "h_revenue": value(m.hydrogen_rev),
        "NPV": value(m.NPV),
        "NPV_ann": value(m.NPV_ann)
    }

    for day in range(num_rep_days):
        results[f'freq_day_{day}'] = value(m.frequency_surrogate[day])

    print("Wind capacity = {} MW".format(value(m.wind_system_capacity) * 1e-3))
    print("PEM capacity = {}MW".format(value(m.pem_system_capacity) * 1e-3))
    print("Plant bid = ${}".format(value(m.pem_bid)))
    print("Plant Elec Revenue Annual = ${}".format(value(m.rev)))
    print("Plant Hydrogen Revenue Annual = ${}".format(value(m.hydrogen_rev)))
    print("Plant Total Revenue Annual = ${}".format(value(m.rev + m.hydrogen_rev)))
    print("Plant NPV = ${}".format(value(m.NPV)))
    print("Plant NPV Annualized = ${}".format(value(m.NPV_ann)))

    print('----------')
    for i in range(num_rep_days):
        print(value(m.frequency_surrogate[i]))
    print('----------')

    print('----------')
    print(sum(value(m.frequency_surrogate[j]) for j in m.dis_set))
    
    return results


def run_design(PEM_bid=None, PEM_size=None):
    default_input_params['pem_mw'] = 317.625
    model, n_rep_days = conceptual_design_dynamic_RE(default_input_params, PEM_bid=PEM_bid, PEM_MW=PEM_size, verbose=False)
    nlp_solver = SolverFactory('ipopt')
    nlp_solver.options['max_iter'] = 8000
    # nlp_solver.options['acceptable_tol'] = 1e-1
    # nlp_solver.options['bound_push'] = 1e-9
    res = nlp_solver.solve(model, tee=True)
    if res.Solver.status != 'ok':
        solve_log = idaeslog.getInitLogger("infeasibility", idaeslog.INFO, tag="properties")
        log_infeasible_constraints(model, logger=solve_log, tol=1e-4, log_expression=True, log_variables=True)
        log_infeasible_bounds(model, logger=solve_log, tol=1e-4)
        return {
        "wind_mw": 0,
        "pem_mw": 0,
        "pem_bid": 0,
        "e_revenue": 0,
        "h_revenue": 0,
        "NPV": 0
    }
    return record_result(model, n_rep_days)

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

    "wind_cap_cost": wind_cap_cost if include_wind_capital_cost else 0,
    "wind_op_cost": wind_op_cost,
    "pem_cap_cost": pem_cap_cost,
    "pem_op_cost": pem_op_cost,
    "pem_var_cost": pem_var_cost
} 


if __name__ == "__main__":
    result = run_design()
    exit()

    import multiprocessing as mp
    from itertools import product

    bids = np.linspace(15, 45, 13)
    sizes = np.linspace(127.05, 423.5, 15)
    inputs = product(bids, sizes)

    with mp.Pool(processes=24) as p:
        res = p.starmap(run_design, inputs)

    df = pd.DataFrame(res)
    df.to_csv(f"surrogate_results_ss_rt_{shortfall}.csv")
