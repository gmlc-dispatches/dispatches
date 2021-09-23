
#This file simply maximizes surrogate revenue without any plant model. 
import pyomo.environ as pe
from pyomo.environ import Var,Expression,Constraint,NonNegativeReals, Objective, value, ConcreteModel, SolverFactory, units, Block,  Param
import rts_surrogates
from pyomo.environ import 

#maximize revenue for surrogate with 5 terms
revenue_rule_5 = rts_surrogates.revenue_rule_5_terms
revenue_rule_all = rts_surrogates.revenue_rule_all_terms
zone_rule_list = [rts_surrogates.hours_zone_0,rts_surrogates.hours_zone_1,rts_surrogates.hours_zone_2,rts_surrogates.hours_zone_3,rts_surrogates.hours_zone_4,rts_surrogates.hours_zone_5,rts_surrogates.hours_zone_6,
rts_surrogates.hours_zone_7,rts_surrogates.hours_zone_8,rts_surrogates.hours_zone_9,rts_surrogates.hours_zone_10]

def create_revenue_model(rule):

    m = pe.ConcreteModel()

    m.pmax = Var(within=NonNegativeReals, bounds=(175,445), initialize=400)
    m.pmin = Expression(expr = 0.3*m.pmax)

    #surrogate market inputs (not technically part of rankine cycle model)
    m.ramp_rate = Var(within=NonNegativeReals, bounds=(48,400), initialize=200)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1,16), initialize=1)
    m.min_down_time = Var(within=NonNegativeReals, bounds=(0.5,32), initialize=1)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=30)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.st_time_hot =  Var(within=NonNegativeReals, bounds=(0.11,1), initialize=1)
    m.st_time_warm =  Var(within=NonNegativeReals, bounds=(0.22,2.5), initialize=1)
    m.st_time_cold =  Var(within=NonNegativeReals, bounds=(0.44,7.5), initialize=1)
    m.st_cst_hot =  Var(within=NonNegativeReals, bounds=(0,85), initialize=40)
    m.st_cst_warm =  Var(within=NonNegativeReals, bounds=(0,120), initialize=40)
    m.st_cst_cold =  Var(within=NonNegativeReals, bounds=(0,150), initialize=40)

    #Revenue surrogate
    m.rev_expr = Expression(rule = rule)

    m.cst_con_1 = Constraint(expr = m.st_time_warm >= m.st_time_hot)
    m.cst_con_2 = Constraint(expr = m.st_time_cold >= m.st_time_warm)
    m.cst_con_3 = Constraint(expr = m.st_cst_warm >= m.st_cst_hot)
    m.cst_con_4 = Constraint(expr = m.st_cst_cold >= m.st_cst_warm)

    m.obj = Objective(expr=-(m.rev_expr))

    return m

m = create_revenue_model(revenue_rule_5)
solver = pe.SolverFactory('ipopt')
status = solver.solve(m, tee=True)

print("Revenue [$MM]: ",value(m.rev_expr))

x = [value(m.pmax),value(m.pmin),value(m.ramp_rate),
    value(m.min_up_time),
    value(m.min_down_time),
    value(m.marg_cst),
    value(m.no_load_cst),
    value(m.st_time_hot),
    value(m.st_time_warm),
    value(m.st_time_cold),
    value(m.st_cst_hot),
    value(m.st_cst_warm),
    value(m.st_cst_cold)]

print(x)

m = create_revenue_model(revenue_rule_all)
solver = pe.SolverFactory('ipopt')
status = solver.solve(m, tee=True)

print("Revenue [$MM]: ",value(m.rev_expr))

x = [value(m.pmax),value(m.pmin),value(m.ramp_rate),
    value(m.min_up_time),
    value(m.min_down_time),
    value(m.marg_cst),
    value(m.no_load_cst),
    value(m.st_time_hot),
    value(m.st_time_warm),
    value(m.st_time_cold),
    value(m.st_cst_hot),
    value(m.st_cst_warm),
    value(m.st_cst_cold)]

print(x)

#
# TEST ZONE SURROGATES
#
for i in range(0,11):
    m = ConcreteModel()

    # Create capex plant
    m.pmax = Var(within=pyo.NonNegativeReals, bounds=(200,450))
    m.pmin = Var(within=pyo.NonNegativeReals, bounds=(0,200))
    m.hours_zone = Expression(rule = zone_rule_list[i])
    m.obj = Objective(expr=(-m.hours_zone))
    m.con_zone1 = Constraint(expr = m.hours_zone <= 8000)
    m.con_zone2 = Constraint(expr = m.hours_zone >= 0)

    opt = pyo.SolverFactory('ipopt')
    opt.solve(m,tee = "True")

    print(pyo.value(m.obj))

pmax = value(m.pmax)
pmin = value(m.pmin)
ramp_rate = 250.0
min_up_time = 3
min_down_time = 5
marg_cst = 5.0
no_load_cst = 1
st_time_hot = 1
st_time_warm = 1
st_time_cold = 1
st_cst_hot = 0
st_cst_warm = 0
st_cst_cold = 0
x = [pmax,pmin,ramp_rate,min_up_time,min_down_time,marg_cst,no_load_cst,st_time_hot,st_time_warm,st_time_cold,st_cst_hot,st_cst_warm,st_cst_cold]

from zone_surrogates.hours_zone_10 import f
f(*x)

