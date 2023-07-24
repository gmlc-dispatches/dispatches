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

import os
import json
from pyomo.common.fileutils import this_file_dir
import omlt
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation
from omlt.io import load_keras_sequential
from tensorflow import keras
from tensorflow.keras import layers
import pyomo.environ as pyo
from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, maximize, units as pyunits, RangeSet
from idaes.core.solvers import get_solver
'''
We will build an optimization problem in this demo.

The object function is the profit = revenue - cost.

The revenue is the NN model from FE case study. It has 2 input variables, storage size and the discharge marginal cost.

The cost is a function of the above variables. I just build a linear function for demo. 
'''

surrogate_dir = os.path.join(this_file_dir(),"FE_case_study")

# load scaling and bounds for NN revenue
with open(os.path.join(surrogate_dir,"FE_revenue_params.json"), 'rb') as f:
    rev_data = json.load(f)

# load keras neural networks
nn_rev = keras.models.load_model(os.path.join(surrogate_dir,"FE_revenue"))

# load keras models and create OMLT NetworkDefinition objects
# Revenue model definition
input_bounds_rev = {i:(rev_data['xmin'][i],rev_data['xmax'][i]) for i in range(len(rev_data['xmin']))}
scaling_object_rev = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                                            factor_inputs=rev_data['xstd_inputs'],
                                            offset_outputs=[rev_data['y_mean']],
                                            factor_outputs=[rev_data['y_std']])

net_rev_defn = load_keras_sequential(nn_rev,scaling_object_rev,input_bounds_rev)

# define model
m = ConcreteModel(name = 'omlt_keras_demo_model')

# define variables
m.storage_size = Var(within=NonNegativeReals, bounds=(15,150), initialize=15)
m.dis_marginal_cost = Var(within=NonNegativeReals, bounds=(39.7,72.7), initialize=71.7)
m.reserve = Var(initialize = 10)
m.max_lmp = Var(initialize = 500)
# m.storage_size.fix(15)
# m.dis_marginal_cost.fix(41.7)
m.reserve.fix(10)
m.max_lmp.fix(500)

inputs = [m.dis_marginal_cost,m.storage_size,m.reserve,m.max_lmp]

# define parameters
m.storage_cost = Param(default = 1e2)    # assume 1e4$/MWh for storage cost

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

# # make rev non-negative, MM$
# m.rev = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

# define obj function
m.obj = Objective(expr = m.rev_surrogate, sense = maximize)
# - m.storage_size*m.storage_cost
# set solver, default = ipopt
nlp_solver = get_solver()
nlp_solver.solve(m, tee=True)
# m.pprint()

print(value(m.storage_size))
print(value(m.dis_marginal_cost))
print(value(m.rev_surrogate))