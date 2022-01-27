import pyomo.environ as pyo
from pyomo.environ import Var,Expression,Constraint,NonNegativeReals, Objective, value, ConcreteModel, SolverFactory, Block,  Param
import os
import pandas as pd
import numpy as np
import pickle
import json
from read_scikit_to_optml import load_scikit_mlp
import omlt
from omlt.neuralnet import NetworkDefinition

# load scaling and bounds
with open("surrogates_neuralnet/prescient_scaling_parameters.json", 'rb') as f:
    data = json.load(f)

xm = data["xm_inputs"]
xstd = data["xstd_inputs"]
x_lower = data["xmin"]
x_upper = data["xmax"]
zm_revenue = data["zm_revenue"]
zstd_revenue = data["zstd_revenue"]
zm_zone_hours = data["zm_zones"]
zstd_zone_hours = data["zstd_zones"]

#provide bounds on the input variable (e.g. from training)
input_bounds = list(zip(x_lower,x_upper))

#####################################################
# TEST REVENUE SURROGATES
#####################################################
def create_revenue_optml_model(nn,fix_nominal_surrogate_inputs=False):

    scaling_object_revenue = optml.OffsetScaling(offset_inputs=xm,
                    factor_inputs=xstd,
                    offset_outputs=[zm_revenue],
                    factor_outputs=[zstd_revenue])

    net = load_scikit_mlp(nn,scaling_object_revenue,input_bounds)
    m = pyo.ConcreteModel()

    m.pmax = Var(within=NonNegativeReals, bounds=(175,450), initialize=300)
    m.pmin_coeff = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)
    m.pmin = Expression(expr = m.pmin_coeff*m.pmax)
    m.ramp_rate = Var(within=NonNegativeReals, bounds=(48,400), initialize=200)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1,16), initialize=1)
    m.min_down_time = Var(within=NonNegativeReals, bounds=(0.5,32), initialize=1)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=5)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.st_time_hot =  Var(within=NonNegativeReals, bounds=(0.11,1), initialize=1)
    m.st_time_warm =  Var(within=NonNegativeReals, bounds=(0.22,2.5), initialize=1)
    m.st_time_cold =  Var(within=NonNegativeReals, bounds=(0.44,7.5), initialize=1)
    m.st_cst_hot =  Var(within=NonNegativeReals, bounds=(0,95), initialize=40)
    m.st_cst_warm =  Var(within=NonNegativeReals, bounds=(0,135), initialize=40)
    m.st_cst_cold =  Var(within=NonNegativeReals, bounds=(0,147), initialize=40)

    #Fix to nominal inputs
    if fix_nominal_surrogate_inputs:
        m.no_load_cst.fix(1.0)
        m.min_up_time.fix(4)
        m.min_down_time.fix(4)
        m.st_time_hot.fix(0.375)
        m.st_time_warm.fix(1.375)
        m.st_time_cold.fix(7.5)
        m.st_cst_hot.fix(94.0)
        m.st_cst_warm.fix(101.5)
        m.st_cst_cold.fix(147.0)
    else:
        #market input constraints
        m.min_dn_multipler = Var(within=NonNegativeReals, bounds=(0.5,2.0), initialize=1.0)
        m.min_dn_time = Constraint(expr = m.min_down_time == m.min_dn_multipler*m.min_up_time)
        m.cst_con_1 = Constraint(expr = m.st_time_warm >= 2*m.st_time_hot)
        m.cst_con_2 = Constraint(expr = m.st_time_cold >= 2*m.st_time_warm)
        m.cst_con_3 = Constraint(expr = m.st_cst_warm >= m.st_cst_hot)
        m.cst_con_4 = Constraint(expr = m.st_cst_cold >= m.st_cst_warm)

    #ramp connects to pmin and pmax
    m.ramp_coeff = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.ramp_limit = Constraint(expr = m.ramp_rate == m.ramp_coeff*(m.pmax - m.pmin))

    m.revenue_surrogate = Var()

    m.nn = optml.OptMLBlock()

    #the neural net contains one input and one output
    # input_set = pyo.Set(initialize = range(n_inputs),ordered=True)
    # pyomo_model.inputs = pyo.Var(input_set)
    # pyomo_model.revenue = pyo.Var()
    m.inputs = [m.pmax,m.pmin,m.ramp_rate,m.min_up_time,m.min_down_time,m.marg_cst,m.no_load_cst,m.st_time_hot,m.st_time_warm,m.st_time_cold,m.st_cst_hot,m.st_cst_warm,m.st_cst_cold]

    formulation = optml.neuralnet.ReducedSpaceContinuousFormulation(net)

    #encodes intermediate neural network variables
    #formulation = optml.neuralnet.FullSpaceContinuousFormulation(net)

    #Revenue surrogate
    #build the formulation on the OptML block
    m.nn.build_formulation(formulation,input_vars=m.inputs, output_vars=[m.revenue_surrogate])
    m.revenue = Expression(expr=0.5*pyo.sqrt(m.revenue_surrogate**2 + 0.001**2) + 0.5*m.revenue_surrogate)
    m.obj = Objective(expr=-(m.revenue))

    return m

def create_zone_optml_model(nn,i,fix_nominal_surrogate_inputs=False):

    scaling_object_zone = optml.OffsetScaling(offset_inputs=xm,
                    factor_inputs=xstd,
                    offset_outputs=[zm_zone_hours[i]],
                    factor_outputs=[zstd_zone_hours[i]])

    net = load_scikit_mlp(nn,scaling_object_zone,input_bounds)
    m = pyo.ConcreteModel()

    m.pmax = Var(within=NonNegativeReals, bounds=(175,450), initialize=300)
    m.pmin_coeff = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)
    m.pmin = Expression(expr = m.pmin_coeff*m.pmax)
    m.ramp_rate = Var(within=NonNegativeReals, bounds=(48,400), initialize=200)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1,16), initialize=1)
    m.min_down_time = Var(within=NonNegativeReals, bounds=(0.5,32), initialize=1)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=5)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.st_time_hot =  Var(within=NonNegativeReals, bounds=(0.11,1), initialize=1)
    m.st_time_warm =  Var(within=NonNegativeReals, bounds=(0.22,2.5), initialize=1)
    m.st_time_cold =  Var(within=NonNegativeReals, bounds=(0.44,7.5), initialize=1)
    m.st_cst_hot =  Var(within=NonNegativeReals, bounds=(0,95), initialize=40)
    m.st_cst_warm =  Var(within=NonNegativeReals, bounds=(0,135), initialize=40)
    m.st_cst_cold =  Var(within=NonNegativeReals, bounds=(0,147), initialize=40)

    #Fix to nominal inputs
    if fix_nominal_surrogate_inputs:
        m.no_load_cst.fix(1.0)
        m.min_up_time.fix(4)
        m.min_down_time.fix(4)
        m.st_time_hot.fix(0.375)
        m.st_time_warm.fix(1.375)
        m.st_time_cold.fix(7.5)
        m.st_cst_hot.fix(94.0)
        m.st_cst_warm.fix(101.5)
        m.st_cst_cold.fix(147.0)
    else:
        #market input constraints
        m.min_dn_multipler = Var(within=NonNegativeReals, bounds=(0.5,2.0), initialize=1.0)
        m.min_dn_time = Constraint(expr = m.min_down_time == m.min_dn_multipler*m.min_up_time)
        m.cst_con_1 = Constraint(expr = m.st_time_warm >= 2*m.st_time_hot)
        m.cst_con_2 = Constraint(expr = m.st_time_cold >= 2*m.st_time_warm)
        m.cst_con_3 = Constraint(expr = m.st_cst_warm >= m.st_cst_hot)
        m.cst_con_4 = Constraint(expr = m.st_cst_cold >= m.st_cst_warm)

    #ramp connects to pmin and pmax
    m.ramp_coeff = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.ramp_limit = Constraint(expr = m.ramp_rate == m.ramp_coeff*(m.pmax - m.pmin))

    m.zone_surrogate = Var()

    m.nn = optml.OptMLBlock()
    m.inputs = [m.pmax,m.pmin,m.ramp_rate,m.min_up_time,m.min_down_time,m.marg_cst,m.no_load_cst,m.st_time_hot,m.st_time_warm,m.st_time_cold,m.st_cst_hot,m.st_cst_warm,m.st_cst_cold]

    formulation = optml.neuralnet.ReducedSpaceContinuousFormulation(net)

    #encodes intermediate neural network variables
    #formulation = optml.neuralnet.FullSpaceContinuousFormulation(net)

    #Revenue surrogate
    #build the formulation on the OptML block
    m.nn.build_formulation(formulation,input_vars=m.inputs, output_vars=[m.zone_surrogate])
    m.zone_hours = Expression(expr=0.5*pyo.sqrt(m.zone_surrogate**2 + 0.001**2) + 0.5*m.zone_surrogate)
    m.obj = Objective(expr=-(m.zone_hours))

    return m

#Maximize neural network revenue
with open('surrogates_neuralnet/cappedRevenueModel.pkl', 'rb') as f:
    nn = pickle.load(f)
m = create_revenue_optml_model(nn,fix_nominal_surrogate_inputs=False)
m.marg_cst.fix(28.0)
#query inputs and outputs, as well as scaled inputs and outputs 
m.nn.inputs_list
m.nn.outputs_list 
m.nn.scaled_inputs_list 
m.nn.scaled_outputs_list
status = pyo.SolverFactory('ipopt').solve(m, tee=True)
for i in range(len(m.inputs)):
    print(pyo.value(m.inputs[i]))
print("Revenue [$MM/year] ",pyo.value(m.revenue))

#Maximize neural network zone hours
zone_hours = []
for i in range(11):
    with open('surrogates_neuralnet/dispatchModelZone{}.pkl'.format(i), 'rb') as f:
        nn = pickle.load(f)
    m = create_zone_optml_model(nn,i,fix_nominal_surrogate_inputs=False)

    status = pyo.SolverFactory('ipopt').solve(m, tee=False)
    zone_hours.append(pyo.value(m.zone_hours))

print("Zone Hours [hr]: ",zone_hours)