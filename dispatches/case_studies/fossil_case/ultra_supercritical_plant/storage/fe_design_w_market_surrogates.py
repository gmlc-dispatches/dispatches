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

"""
This script runs the design case study for the fossils case using
market surrogates. It uses the multiperiod model class for the integrated
power plant with energy storage. The revenue and dispatch surrogates are in
keras. This implementation is pinned to the following version of the respective
packages. OMLT is v1.0, ONNX is v1.12.0, TensorFlow is v1.19.4.
"""

__author__ = "Naresh Susarla"


import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from pyomo.common.fileutils import this_file_dir
from pyomo.environ import (
    ConcreteModel,
    Param,
    SolverFactory,
    Var, 
    NonNegativeReals, 
    units as pyunits, 
    Expression, 
    Constraint,
    Objective, 
    sqrt,
)
import idaes.logger as idaeslog
from idaes.apps.grid_integration.multiperiod.multiperiod import (
    MultiPeriodModel)
from dispatches.case_studies.fossil_case.ultra_supercritical_plant.storage.\
    multiperiod_integrated_storage_usc import (create_usc_model,
                                               usc_custom_init,
                                               # usc_unfix_dof,
                                               get_usc_link_variable_pairs)

import omlt  # omlt can encode the neural networks in Pyomo
from omlt.neuralnet import FullSpaceNNFormulation
from omlt.io import load_keras_sequential
# from pyomo.repn.plugins.nl_writer import _activate_nl_writer_version
# _activate_nl_writer_version(2)

# For plots
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rc('font', size=24)
plt.rc('axes', titlesize=24)
scaling_obj = 1
scaling_factor = 1


NUM_REP_DAYS = 20               # Number of clusters/representative days
n_time_points = 24
# path for folder that has surrogate models
surrogate_dir = os.path.join(this_file_dir(), "FE_surrogates")

# ================= Read the cluster centers (dispatch representative days) ====================
with open(os.path.join(surrogate_dir, "FE_result_400years_20clusters_OD.json"), 'r') as f:
    cluster_results = json.load(f)
cluster_center = np.array(cluster_results['model_params']['cluster_centers_'])
cluster_center = cluster_center.reshape(20, 24)[0:NUM_REP_DAYS]
# cluster_condensed = cluster[0:2, :]
# add zero/full capacity days to the clustering results. 
full_days = np.array([np.ones(24)])
zero_days = np.array([np.zeros(24)])

# corresponds to the ws, ws[0] is for zero cf days and ws[31] is for full cd days.
cluster_centers = np.concatenate((zero_days, cluster_center, full_days), axis=0)
# ===============================================================================================

# ================ Get input-output scaling factors for dispatch frequency ======================
with open(os.path.join(surrogate_dir, "FE_20clusters_dispatch_frequency_params.json"), 'rb') as f:
    dispatch_data = json.load(f)

# Dispatch frequency surrogate
input_bounds_dispatch = {
    i: (dispatch_data['xmin'][i], dispatch_data['xmax'][i]) 
    for i in range(len(dispatch_data['xmin']))
}
scaling_object_dispatch = omlt.OffsetScaling(
    offset_inputs=dispatch_data['xm_inputs'],
    factor_inputs=dispatch_data['xstd_inputs'],
    offset_outputs=dispatch_data['ws_mean'],
    factor_outputs=dispatch_data['ws_std'],
)

# load keras neural networks for weights 
nn_dispatch = keras.models.load_model(os.path.join(surrogate_dir, "FE_20clusters_dispatch_frequency"))
net_dispatch_defn = load_keras_sequential(nn_dispatch, scaling_object_dispatch, input_bounds_dispatch)
# ==================================================================================================

# ================ Get input-output scaling factors for revenue ====================================
with open(os.path.join(surrogate_dir, "FE_revenue_params.json"), 'rb') as f:
    revenue_data = json.load(f)

input_bounds_revenue = {
    i: (revenue_data['xmin'][i], revenue_data['xmax'][i]) 
    for i in range(len(revenue_data['xmin']))
}
scaling_object_revenue = omlt.OffsetScaling(
    offset_inputs=revenue_data["xm_inputs"],
    factor_inputs=revenue_data["xstd_inputs"],
    offset_outputs=[revenue_data["y_mean"]],
    factor_outputs=[revenue_data["y_std"]],
)

nn_revenue = keras.models.load_model(os.path.join(surrogate_dir, "FE_revenue"))
revenue_defn = load_keras_sequential(nn_revenue, scaling_object_revenue, input_bounds_revenue)
# ====================================================================================================

def _get_dispatch_capacity_factors():
    
    file_name = 'FE_dispatch_95_5_median_separate.json'
    # file_name = 'FE_dispatch_95_5_median_separate_3_day.json'

    with open(os.path.join(surrogate_dir, file_name), 'rb') as f:
        sep_data = json.load(f)

    # this for no sub scenario model.
    full_day_gen_cf = [1.0]*24
    full_day_stor_cf = [0.0]*24
    
    zero_day_gen_cf = [0.0]*24
    zero_day_stor_cf = [0.0]*24
    
    cluster_cf = sep_data['median_dispatch']
    
    gen_cf = []
    stor_cf = []

    # append cf for zero days
    gen_cf.append(zero_day_gen_cf)
    stor_cf.append(zero_day_stor_cf)

    # append cf for rep days
    for index, value in enumerate(cluster_cf):
        gen_cf.append(cluster_cf[str(index)][0])
        stor_cf.append(cluster_cf[str(index)][1])

    # append cf for full day
    gen_cf.append(full_day_gen_cf)
    stor_cf.append(full_day_stor_cf)

    return gen_cf, stor_cf

def usc_unfix_dof(m):
    # Unfix data
    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Unfix storage system data
    m.fs.ess_hp_split.split_fraction[0, "to_hxc"].unfix()
    m.fs.ess_bfp_split.split_fraction[0, "to_hxd"].unfix()
    for salt_hxc in [m.fs.hxc]:
        salt_hxc.shell_inlet.unfix()
        salt_hxc.tube_inlet.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxc.area.unfix()  # 1 DOF

    for salt_hxd in [m.fs.hxd]:
        salt_hxd.tube_inlet.unfix()
        salt_hxd.shell_inlet.flow_mass.unfix()  # kg/s, 1 DOF
        salt_hxd.area.unfix()  # 1 DOF

    for unit in [m.fs.cooler]:
        unit.inlet.unfix()
    m.fs.cooler.outlet.enth_mol[0].unfix()  # 1 DOF

    # Fix storage heat exchangers area and salt temperatures
    # m.fs.hxc.area.fix(1904)
    # m.fs.hxd.area.fix(2830)
    m.fs.hxc.tube_outlet.temperature[0].fix(831)
    m.fs.hxd.shell_inlet.temperature[0].fix(831)
    m.fs.hxd.shell_outlet.temperature[0].fix(513.15)
    
def build_design_model_w_surrogates(n_rep_days, reserve=10, max_lmp=500):

    # Concrete model
    m = ConcreteModel(name='FE_design_model_w_surrogates')

    # Inputs to the frequency surrogates: discharge marginal cost, storage size, reserve %, max lmp
    m.discharge_marginal_cost = Var(
        within=NonNegativeReals,
        bounds=(40, 80),
        initialize=70,
        doc="Marginal cost at which storage will be discharged",
    )
    m.storage_size = Var(
        within=NonNegativeReals,
        bounds=(15, 150),
        initialize=150,
        doc="Size of storage in MW",
    )
    m.reserve = Var(
        within=NonNegativeReals,
        doc="Percentage of reserves",
    )
    m.max_lmp = Var(
        within=NonNegativeReals,
        doc="Maximum LMP",
    )
    m.reserve.fix(reserve)
    m.max_lmp.fix(max_lmp)  

    # Add NN surrogates for dispatch frequency to the model using omlt
    m.fe_dispatch = omlt.OmltBlock()
    m.fe_dispatch.build_formulation(FullSpaceNNFormulation(net_dispatch_defn))

    # Add NN surrogate for revenue to the model
    m.fe_revenue = omlt.OmltBlock()
    m.fe_revenue.build_formulation(FullSpaceNNFormulation(revenue_defn))

    inputs = [m.discharge_marginal_cost, m.storage_size, m.reserve, m.max_lmp]  

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_dispatch(blk, i):
        return inputs[i] == blk.fe_dispatch.inputs[i]

    @m.Constraint(range(len(inputs)))
    def input_variable_equality_revenue(blk, i):
        return inputs[i] == blk.fe_revenue.inputs[i]
    
    m.set_days = [i for i in range(n_rep_days)]

    m.electricity_revenue = Var(within=NonNegativeReals, initialize=1e9)
    m.weights = Var(
        m.set_days, 
        within=NonNegativeReals,
        initialize={x: 1 / n_rep_days for x in m.set_days},
    )
    m.weights_non_neg = Var(
        m.set_days, 
        within=NonNegativeReals,
        initialize={x: 1 / n_rep_days for x in m.set_days},
    )

    # Convert any negative weights to postive weights.
    @m.Constraint(m.set_days)
    def non_neg_weights_definition(blk, i):
        return (
            blk.weights_non_neg[i] == 0.5 * sqrt(blk.fe_dispatch.outputs[i]**2 + 0.001**2) + 0.5 * blk.fe_dispatch.outputs[i]
        )

    # Re-scale weights so that the add up to one
    @m.Constraint(m.set_days)
    def weights_definition(blk, i):
        return (
            blk.weights[i] * sum(blk.weights_non_neg[j] for j in m.set_days) == blk.weights_non_neg[i]
        )

    # Revenue from the surrogates
    m.revenue_surrogate = Constraint(expr=m.electricity_revenue == m.fe_revenue.outputs[0])

    # Create the multiperiod model object. You can pass arguments to your
    # "process_model_func" for each time period using a dict of dicts as
    # shown here.  In this case, it is setting up empty dictionaries for
    # each time period.
    m.mp_fe_model = MultiPeriodModel(
        n_time_points=n_time_points,
        set_days=m.set_days,
        process_model_func=create_usc_model,
        initialization_func=usc_custom_init,
        unfix_dof_func=usc_unfix_dof,
        linking_variable_func=get_usc_link_variable_pairs,
        flowsheet_options={"pmin": None,
                           "pmax": None},
        use_stochastic_build=True,
        outlvl=idaeslog.INFO,
        )

    # Retrieve active process blocks (i.e. time blocks)
    blks = [m.mp_fe_model.period[t, d] for (t, d) in m.mp_fe_model.set_period]
    
    # Compute operating cost
    for blk in blks:
        blk.operating_cost = Expression(
            expr=(
                (blk.fs.operating_cost
                 + blk.fs.plant_fixed_operating_cost
                 + blk.fs.plant_variable_operating_cost) / (365 * 24)
            )
        )

    # Compute the total operating cost for all rep days
    m.total_operating_cost = Expression(
    expr=sum(
        m.weights[d] * 366 * m.mp_fe_model.period[t, d].operating_cost
        for (t, d) in m.mp_fe_model.set_period
    ))

    # Add the dispatch constaint on the net power using the capcity factors
    # The power must be scaled using the capacity factor, Pmin, Pmax, Pstorage
    m.plant_pmin = Param(
        initialize=284,
        mutable=False,
        units=pyunits.MW,
        doc='Pmin for thermal generator alone')
    m.plant_pmax = Param(
        initialize=436,
        mutable=False,
        units=pyunits.MW,
        doc='Pmax for thermal generator alone')

    # Get dispatch capcity factors from json
    cf_plant, cf_storage = _get_dispatch_capacity_factors()

    @m.Constraint(m.mp_fe_model.set_period)
    def power_dispatch_constraint(blk, t, d):
        return (
            blk.mp_fe_model.period[t, d].fs.net_power >=
            m.plant_pmin + (m.plant_pmax - m.plant_pmin) * cf_plant[d][t-1] +
            m.storage_size * cf_storage[d][t-1]
        )

    m.hxc_area = Var(
        within=NonNegativeReals,
        bounds=(0, 5000),
        initialize=1904,
        doc="area of charge heat exchanger",
    )
    m.hxd_area = Var(
        within=NonNegativeReals,
        bounds=(0, 5000),
        initialize=2830,
        doc="area of discharge heat exchanger",
    )
    @m.Constraint(m.mp_fe_model.set_period)
    def charge_hx_area_constraint(blk, t, d):
        return (
            blk.mp_fe_model.period[t, d].fs.hxc.area ==
            m.hxc_area
        )
    @m.Constraint(m.mp_fe_model.set_period)
    def discharge_hx_area_constraint(blk, t, d):
        return (
            blk.mp_fe_model.period[t, d].fs.hxd.area ==
            m.hxd_area
        )

    # The objective is minimize: cost - revenue
    m.obj = Objective(expr=m.total_operating_cost - m.electricity_revenue)

    # Initial state for the power plant the salt tank are fixed
    for d in m.set_days:
        # Starting from a fully charged state for each representative day
        m.mp_fe_model.period[1, d].fs.previous_salt_inventory_hot.fix(6739292-1103053.48)
        m.mp_fe_model.period[1, d].fs.previous_salt_inventory_cold.fix(1103053.48)
        m.mp_fe_model.period[1, d].fs.previous_power.fix(447.66)

    # Plot results
    opt = SolverFactory('ipopt')
    opt.solve(m, tee=True)

    return m



if __name__ == "__main__":

    n_rep_days = NUM_REP_DAYS+2 
    # gen_cf, stor_cf = _get_dispatch_capacity_factors()
    m = build_design_model_w_surrogates(n_rep_days,
                                        reserve=10,
                                        max_lmp=500)

