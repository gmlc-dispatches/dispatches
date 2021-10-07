#This file simply maximizes surrogate revenue without any plant model. 
from pyomo.environ import Var,Expression,Constraint,NonNegativeReals, Objective, value, ConcreteModel, SolverFactory, units, Block,  Param
import pyomo.environ as pyo
from surrogates_alamo import alamo_surrogates

#maximize revenue for surrogate with 5 terms
revenue_rule_5 = alamo_surrogates.revenue_rule_5_terms
revenue_rule_all = alamo_surrogates.revenue_rule_all_terms
zone_rule_list = [alamo_surrogates.hours_zone_0,alamo_surrogates.hours_zone_1,alamo_surrogates.hours_zone_2,alamo_surrogates.hours_zone_3,alamo_surrogates.hours_zone_4,alamo_surrogates.hours_zone_5,alamo_surrogates.hours_zone_6,
alamo_surrogates.hours_zone_7,alamo_surrogates.hours_zone_8,alamo_surrogates.hours_zone_9,alamo_surrogates.hours_zone_10]

#####################################################
# REVENUE SURROGATES
#####################################################
def create_revenue_alamo_model(revenue_rule,fix_nominal_surrogate_inputs=False):

    m = ConcreteModel()

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

    #Revenue surrogate
    m.rev_surrogate = Expression(rule = revenue_rule)
    m.rev_expr = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

    #todo: pmin multiplier
    m.ramp_coeff = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.ramp_limit = Constraint(expr = m.ramp_rate == m.ramp_coeff*(m.pmax - m.pmin))

    m.obj = Objective(expr=-(m.rev_expr))

    return m

#####################################################
# ZONE HOUR SURROGATES
#####################################################
def create_zone_alamo_model(zone_rule,fix_nominal_surrogate_inputs=False):

    m = ConcreteModel()

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

    #Revenue surrogate
    m.zone_surrogate = Expression(rule = zone_rule)

    #todo: pmin multiplier
    m.ramp_coeff = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.ramp_limit = Constraint(expr = m.ramp_rate == m.ramp_coeff*(m.pmax - m.pmin))

    m.obj = Objective(expr=-(m.zone_surrogate))

    return m

#####################################################
# TEST REVENUE SURROGATES
#####################################################
#Test the surrogate containing 5 input variables
m = create_revenue_alamo_model(revenue_rule_5)
solver = SolverFactory('ipopt')
status = solver.solve(m, tee=False)

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

print("Revenue 5 terms inputs: ", x)

#Test the surrogate containing 13 input variables
m = create_revenue_alamo_model(revenue_rule_all)
solver = SolverFactory('ipopt')
status = solver.solve(m, tee=False)

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

print("Revenue 13 terms inputs: ", x)

######################################################
# TEST ZONE SURROGATES
######################################################
zone_hours = []
for i in range(0,11):
    m = create_zone_alamo_model(zone_rule_list[i])

    opt = pyo.SolverFactory('ipopt')
    opt.solve(m,tee = False)

    print(-pyo.value(m.obj))
    zone_hours.append(-pyo.value(m.obj))

print(zone_hours)
