from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective
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
from revenue_rule import revenue_rule
import zone_rules

zone_rule_list = [zone_rules.hours_zone_0,zone_rules.hours_zone_1,zone_rules.hours_zone_2,zone_rules.hours_zone_3,zone_rules.hours_zone_4,zone_rules.hours_zone_5,zone_rules.hours_zone_6,
zone_rules.hours_zone_7,zone_rules.hours_zone_8,zone_rules.hours_zone_9,zone_rules.hours_zone_10]

def stochastic_surrogate_optimization_problem(heat_recovery=False,
                                    p_upper_bound=500,
                                    capital_payment_years=5,
                                    plant_lifetime=20):


    m = ConcreteModel()

    # Create capex plant
    m.cap_fs = create_model(heat_recovery=heat_recovery)
    m.cap_fs = set_inputs(m.cap_fs)
    m.cap_fs = initialize_model(m.cap_fs)
    m.cap_fs = close_flowsheet_loop(m.cap_fs)
    m.cap_fs = add_capital_cost(m.cap_fs)

    # capital cost (M$/yr)
    cap_expr = m.cap_fs.fs.capital_cost*1e6/capital_payment_years

    pmin = 0.3*m.cap_fs.fs.net_cycle_power_output
    pmax = m.cap_fs.fs.net_cycle_power_output

    m.rev_expr = Expression(rule = revenue_rule)

    # Create opex plant
    op_expr = 0

    n_zones = 11
    m.zone_weight_expr = np.zeros(n_zones)
    zone_outputs = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #Create a surrogate for each zone
    for i in range(len(zones)):

        print()
        print("Creating instance ", i)

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
        op_fs.zone_hours = Expression(rule = zone_rule_list[i])

        #We should avoid surrogate outputs that go nuts
        m.constrain_zone_1 = Constraint(expr = op_fs.zone_hours <= 8000)
        m.constrain_zone_2 = Constraint(expr = op_fs.zone_hours >= 0)

        op_expr += op_fs.zone_hours*op_fs.fs.operating_cost

        #Satisfy demand for this zone. Uses design pmax and pmin.
        zone_output = zone_outputs[i]
        op_fs.fs.eq_fix_power = Constraint(expr=op_fs.fs.net_cycle_power_output == zone_output*(m.pmax-m.pmin) + m.pmin)

        op_fs.fs.boiler.inlet.flow_mol[0].unfix()

        # Set bounds for the flow
        op_fs.fs.boiler.inlet.flow_mol[0].setlb(1)

        setattr(m, 'zone_{}'.format(i), op_fs)

    # Expression for total cap and op cost - $
    m.total_cost = Expression(
        expr=plant_lifetime*op_expr + capital_payment_years*cap_expr)

    # Expression for total revenue
    m.total_revenue = Expression(
        expr=plant_lifetime*m.rev_expr)

    # Objective $
    m.obj = Objective(
        expr=-(m.total_revenue - m.total_cost))

    # Unfixing the boiler inlet flowrate for capex plant
    m.cap_fs.fs.boiler.inlet.flow_mol[0].unfix()

    # Setting bounds for the capex plant flowrate
    m.cap_fs.fs.boiler.inlet.flow_mol[0].setlb(5)
    # m.cap_fs.fs.boiler.inlet.flow_mol[0].setub(25000)

    # Setting bounds for net cycle power output for the capex plant
    m.cap_fs.fs.eq_min_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output >= 0)

    m.cap_fs.fs.eq_max_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output <=
        p_upper_bound*1e6)

    return m


#zone_power_min is an expression that puts power in an interval
# op_fs.fs.eq_min_power = Constraint(
#     expr=op_fs.fs.net_cycle_power_output >= zone_power_min[zone])
#
# op_fs.fs.eq_max_power = Constraint(
#     expr=op_fs.fs.net_cycle_power_output <= zone_power_max[zone])
# op_fs.fs.eq_max_power = Constraint(
#     expr=op_fs.fs.net_cycle_power_output <= zone_power_max[zone])

# # only if power demand is given
# if power_demand is not None:
#     op_fs.fs.eq_max_produced = Constraint(
#         expr=op_fs.fs.net_cycle_power_output <= power_demand[i]*1e6)


#zone output depends on design pmin and pmax

#op_fs.zone_output = Expression(expr = zone_output*op_fs.fs.net_cycle_power_output - pmin)

# op_fs.fs.eq_min_power = Constraint(
#     expr=op_fs.fs.net_cycle_power_output >=
#     pmin)
# # operating P_max = design P_max
# op_fs.fs.eq_max_power = Constraint(
#     expr=op_fs.fs.net_cycle_power_output <=
#     pmax)


#Scenarios define operating zones
