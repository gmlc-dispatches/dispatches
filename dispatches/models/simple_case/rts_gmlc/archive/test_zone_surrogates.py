from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective

import pyomo.environ as pyo

import zone_rules

zone_rule_list = [zone_rules.hours_zone_0,zone_rules.hours_zone_1,zone_rules.hours_zone_2,zone_rules.hours_zone_3,zone_rules.hours_zone_4,zone_rules.hours_zone_5,zone_rules.hours_zone_6,
zone_rules.hours_zone_7,zone_rules.hours_zone_8,zone_rules.hours_zone_9,zone_rules.hours_zone_10]

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
