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

from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
from idaes.power_generation.costing.power_plant_costing import get_PP_costing
# Import steam property package
from idaes.generic_models.properties.iapws95 import htpx, Iapws95ParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.util import get_solver
import idaes.logger as idaeslog
import pyomo.environ as pyo

#surrogate functions from alamo
# from revenue_rule import revenue_rule
# import zone_rules
import rts_surrogates

revenue_rule = rts_surrogates.revenue_rule
zone_rule_list = [rts_surrogates.hours_zone_0,rts_surrogates.hours_zone_1,rts_surrogates.hours_zone_2,rts_surrogates.hours_zone_3,rts_surrogates.hours_zone_4,rts_surrogates.hours_zone_5,rts_surrogates.hours_zone_6,
rts_surrogates.hours_zone_7,rts_surrogates.hours_zone_8,rts_surrogates.hours_zone_9,rts_surrogates.hours_zone_10]

#TODO: Make this easier to modify
n_zones = len(zone_rule_list)
zone_outputs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# zone_map = {0.0:0,0.1:1,0.2:2,0.3:3,0.4:4,0.5:5,0.6:6,0.7:7,0.8:8,0.9:9,1.0:10}


def stochastic_surrogate_optimization_problem(heat_recovery=False,
                                    p_lower_bound=10,
                                    p_upper_bound=500,
                                    capital_payment_years=5,
                                    plant_lifetime=20,
                                    zones = [1,2,3,4,5,6,7,8,9,10]):

    m = ConcreteModel()

    # Create capex plant
    m.cap_fs = create_model(heat_recovery=heat_recovery)
    m.cap_fs = set_inputs(m.cap_fs)
    m.cap_fs = initialize_model(m.cap_fs)
    m.cap_fs = close_flowsheet_loop(m.cap_fs)
    m.cap_fs = add_capital_cost(m.cap_fs)

    # capital cost (M$/yr)
    cap_expr = m.cap_fs.fs.capital_cost/capital_payment_years

    #m.pmax = m.cap_fs.value()

    m.pmax = Expression(expr = 1.0*m.cap_fs.fs.net_cycle_power_output*1e-6)
    m.pmin = Expression(expr = 0.3*m.pmax)
    # m.pmin = Var(within=NonNegativeReals, bounds=(0,200), initialize=200)
    # m.pmin_lower = Constraint(expr = m.pmin >= 0.3*m.pmax)
    # m.pmin_upper = Constraint(expr = m.pmin <= 1.0*m.pmax)

    #surrogate market inputs (not technically part of rankine cycle model)
    m.ramp_rate = Var(within=NonNegativeReals, bounds=(48,400), initialize=200)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1,16), initialize=1)
    m.min_down_time = Var(within=NonNegativeReals, bounds=(0.5,32), initialize=1)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=15)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.st_time_hot =  Var(within=NonNegativeReals, bounds=(0.11,1), initialize=1)
    m.st_time_warm =  Var(within=NonNegativeReals, bounds=(0.22,2.5), initialize=1)
    m.st_time_cold =  Var(within=NonNegativeReals, bounds=(0.44,7.5), initialize=1)
    m.st_cst_hot =  Var(within=NonNegativeReals, bounds=(0,85), initialize=40)
    m.st_cst_warm =  Var(within=NonNegativeReals, bounds=(0,120), initialize=40)
    m.st_cst_cold =  Var(within=NonNegativeReals, bounds=(0,150), initialize=1)

    #Revenue surrogate
    m.rev_expr = Expression(rule = revenue_rule)

    # Create expression for opex plant
    op_expr = 0
    #Create a surrogate for each zone
    # for i in range(len(zone_outputs)):
    for i in zones:
        print()
        print("Creating instance ", i)
        zone_output = zone_outputs[i]
        op_fs = create_model(heat_recovery=heat_recovery)

        # Set model inputs for the capex and opex plant
        op_fs = set_inputs(op_fs)

        # Initialize the capex and opex plant
        op_fs = initialize_model(op_fs)

        # Closing the loop in the flowsheet
        op_fs = close_flowsheet_loop(op_fs)

        #This will be the scenario
        op_fs = add_operating_cost(op_fs)

        #weights come from surrogate
        op_fs.pmax = Expression(expr = m.pmax)
        op_fs.pmin = Expression(expr = m.pmin)
        op_fs.ramp_rate = Expression(expr = m.ramp_rate)
        op_fs.min_up_time = Expression(expr = m.min_up_time)
        op_fs.min_down_time = Expression(expr = m.min_down_time)
        op_fs.marg_cst =  Expression(expr = m.marg_cst)
        op_fs.no_load_cst =  Expression(expr = m.no_load_cst)
        op_fs.st_time_hot =  Expression(expr = m.st_time_hot)
        op_fs.st_time_warm =  Expression(expr = m.st_time_warm)
        op_fs.st_time_cold =  Expression(expr = m.st_time_cold)
        op_fs.st_cst_hot = Expression(expr = m.st_cst_hot)
        op_fs.st_cst_warm =  Expression(expr = m.st_cst_warm)
        op_fs.st_cst_cold =  Expression(expr = m.st_cst_cold)

        op_fs.zone_hours_surrogate = Expression(rule = zone_rule_list[i])
        # #smooth max (avoids negative weights)
        op_fs.zone_hours = Expression(expr =  0.5*pyo.sqrt(op_fs.zone_hours_surrogate**2 + 0.001**2) + 0.5*op_fs.zone_hours_surrogate)
        op_expr += op_fs.zone_hours*op_fs.fs.operating_cost

        #Satisfy demand for this zone. Uses design pmax and pmin.
        # if zone_output == 0: #if 'off', no power output (or low output for numeric stability)
        #     op_fs.fs.eq_fix_power = Constraint(expr=op_fs.fs.net_cycle_power_output <= 0.1)
        #     op_fs.fs.boiler.inlet.flow_mol[0].setlb(0)
        # else:
        op_fs.fs.eq_fix_power = Constraint(expr=op_fs.fs.net_cycle_power_output*1e-6 == zone_output*(m.pmax-m.pmin) + m.pmin)
        #op_fs.fs.eq_fix_power1 = Constraint(expr=op_fs.fs.net_cycle_power_output*1e-6 <= zone_outputs[i]*(m.pmax-m.pmin) + m.pmin)
        #op_fs.fs.eq_fix_power2 = Constraint(expr=op_fs.fs.net_cycle_power_output*1e-6 >= zone_outputs[i-1]*(m.pmax-m.pmin) + m.pmin)

        op_fs.fs.boiler.inlet.flow_mol[0].setlb(5)
        op_fs.fs.boiler.inlet.flow_mol[0].unfix()
        setattr(m, 'zone_{}'.format(i), op_fs)

    # Expression for total cap and op cost - $
    m.total_cost = Expression(
        expr=plant_lifetime*op_expr + capital_payment_years*cap_expr)

    # Expression for total revenue
    m.total_revenue = Expression(
        expr=plant_lifetime*m.rev_expr)

    # Objective $
    m.obj = Objective(
        expr=-(m.total_revenue - m.total_cost)/1e9)

    # Unfixing the boiler inlet flowrate for capex plant
    m.cap_fs.fs.boiler.inlet.flow_mol[0].unfix()

    # Setting bounds for the capex plant flowrate
    m.cap_fs.fs.boiler.inlet.flow_mol[0].setlb(5)
    # m.cap_fs.fs.boiler.inlet.flow_mol[0].setub(25000)

    # Setting bounds for net cycle power output for the capex plant
    m.cap_fs.fs.eq_min_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output >= p_lower_bound*1e6)

    m.cap_fs.fs.eq_max_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output <= p_upper_bound*1e6)

    return m
