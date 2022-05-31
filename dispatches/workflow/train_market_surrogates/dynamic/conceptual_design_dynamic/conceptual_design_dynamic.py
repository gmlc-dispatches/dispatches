# conceptual_design_problem_dynamic formation 2, only use timeseries clustering to cluster dispatch data

# import packages
#the rankine cycle is a directory above this one, so modify path
from pyomo.common.fileutils import this_file_dir
import sys, os, json
sys.path.append(os.path.join(this_file_dir(),"../../../models/simple_case"))

from simple_rankine_cycle import *

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

surrogate_dir = os.path.join()

# load scaling and bounds for NN surrogates (revenue and # of startups)

with open(os.path.join(surrogate_dir,"training_parameters_revenue.json"), 'rb') as f:
    rev_data = json.load(f)

with open(os.path.join(surrogate_dir,"training_parameters_nstartups.json"), 'rb') as f:
    nstartups_data = json.load(f)

# load scikit neural networks
with open(os.path.join(surrogate_dir,'scikit_revenue.pkl'), 'rb') as f:
    nn_revenue = pickle.load(f)

with open(os.path.join(surrogate_dir,'scikit_nstartups.pkl'), 'rb') as f:
    nn_nstartups = pickle.load(f)


#load scikit models and create OMLT NetworkDefinition objects
#Revenue model definition
input_bounds_rev = list(zip(rev_data['xmin'],rev_data['xmax']))
scaling_object_revenue = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                factor_inputs=rev_data['xstd_inputs'],
                offset_outputs=[rev_data['zm_revenue']],
                factor_outputs=[rev_data['zstd_revenue']])
net_rev_defn = load_scikit_mlp(nn_revenue,scaling_object_revenue,input_bounds_rev)


#Nstartup model definition
input_bounds_nstartups = list(zip(nstartups_data['xmin'],nstartups_data['xmax']))
scaling_object_nstartups = omlt.OffsetScaling(offset_inputs=nstartups_data['xm_inputs'],
                factor_inputs=nstartups_data['xstd_inputs'],
                offset_outputs=[nstartups_data['zm_nstartups']],
                factor_outputs=[nstartups_data['zstd_nstartups']])
net_nstartups_defn = load_scikit_mlp(nn_nstartups,scaling_object_nstartups,input_bounds_nstartups)



# placeholder for the dispatch frequency surrogate
# fix the number of representative days = 30 
ws = []

def conceptual_design_dynamic_formation2(
	heat_recovery=False,
    calc_boiler_eff=False,
    p_lower_bound=10,
    p_upper_bound=500,
    capital_payment_years=5,
    plant_lifetime=20,
    coal_price=51.96):

# the above rankine cycle model parameter is from 'model_neuralnet_surrogate.py' in Jordan's repo

# for each scenario 
	for w in ws:



