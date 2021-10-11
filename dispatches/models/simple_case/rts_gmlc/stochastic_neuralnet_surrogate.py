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

from read_scikit_to_optml import load_scikit_mlp
import json
import pickle
import optml
from optml.neuralnet import NetworkDefinition

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

with open('surrogates_neuralnet/cappedRevenueModel.pkl', 'rb') as f:
    nn_revenue = pickle.load(f)

#provide bounds on the input variables (e.g. from training)
input_bounds = list(zip(x_lower,x_upper))

#Revenue model definition
scaling_object_revenue = optml.OffsetScaling(offset_inputs=xm,
                factor_inputs=xstd,
                offset_outputs=[zm_revenue],
                factor_outputs=[zstd_revenue])
net_rev_defn = load_scikit_mlp(nn_revenue,scaling_object_revenue,input_bounds)


#Zone model definitions
zone_models = []
zone_defns = []
for i in range(11):
    with open('surrogates_neuralnet/dispatchModelZone{}.pkl'.format(i), 'rb') as f:
        nn_zone = pickle.load(f)
    zone_models.append(nn_zone)
    scaling_object_zone = optml.OffsetScaling(offset_inputs=xm,
    factor_inputs=xstd,
    offset_outputs=[zm_zone_hours[i]],
    factor_outputs=[zstd_zone_hours[i]])
    net_zone_defn = load_scikit_mlp(nn_zone,scaling_object_zone,input_bounds)
    zone_defns.append(net_zone_defn)

#Denote the scaled power output for each of the 10 zones (0 corresponds to pmin, 1.0 corresponds to pmax)
zone_outputs = [0.0,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.0]

def stochastic_surrogate_nn_optimization_problem(
                                    heat_recovery=False,
                                    calc_boiler_eff=False,
                                    p_lower_bound=10,
                                    p_upper_bound=500,
                                    capital_payment_years=5,
                                    plant_lifetime=20,
                                    include_zone_off = True,
                                    fix_nominal_surrogate_inputs = False
                                    ):

    m = ConcreteModel()

    # Create capex plant
    m.cap_fs = create_model(
        heat_recovery=heat_recovery,
        capital_fs=True, calc_boiler_eff=False)
    m.cap_fs = set_inputs(m.cap_fs)
    m.cap_fs = initialize_model(m.cap_fs)
    m.cap_fs = close_flowsheet_loop(m.cap_fs)
    m.cap_fs = add_capital_cost(m.cap_fs)

    # capital cost (M$/yr)
    cap_expr = m.cap_fs.fs.capital_cost/capital_payment_years

    #pmax and pmin
    m.pmax = Expression(expr = 1.0*m.cap_fs.fs.net_cycle_power_output*1e-6)
    m.pmin_coeff = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)
    m.pmin = Expression(expr = m.pmin_coeff*m.pmax)
    #possibly fix pmin multiplier given option
    #m.pmin_coeff.fix(0.3)

    #surrogate market inputs (not technically part of rankine cycle model but are used in market model)
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


    m.ramp_coeff = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.ramp_limit = Constraint(expr = m.ramp_rate == m.ramp_coeff*(m.pmax - m.pmin))

    m.inputs = [m.pmax,m.pmin,m.ramp_rate,m.min_up_time,m.min_down_time,m.marg_cst,m.no_load_cst,m.st_time_hot,m.st_time_warm,
    m.st_time_cold,m.st_cst_hot,m.st_cst_warm,m.st_cst_cold]

    #Revenue OptML surrogate
    m.rev_surrogate = Var()
    m.nn = optml.OptMLBlock()

    #build the formulation on the OptML block
    formulation = optml.neuralnet.ReducedSpaceContinuousFormulation(net_rev_defn)
    m.nn.build_formulation(formulation,input_vars=m.inputs, output_vars=[m.rev_surrogate])
    m.revenue = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

    ############################################
    #'off' zone
    ############################################
    if include_zone_off:
        off_fs = Block()
        off_fs.fs = Block()
        off_fs.fs.operating_cost = 0.0

        #Zone hour surrogate     
        #BUG: off_fs.nn.scaled_inputs_list is not constructed if we put the OptML block on a sublock.
        #This only works if the optml blocks are on the pyomo concrete model
        off_fs.zone_hours_surrogate = Var()
        m.nn_off = optml.OptMLBlock()

        #build the formulation on the OptML block
        formulation = optml.neuralnet.ReducedSpaceContinuousFormulation(zone_defns[0])
        m.nn_off.build_formulation(formulation,input_vars=m.inputs, output_vars=[off_fs.zone_hours_surrogate])
        off_fs.zone_hours = Expression(expr=0.5*pyo.sqrt(off_fs.zone_hours_surrogate**2 + 0.001**2) + 0.5*off_fs.zone_hours_surrogate)
        setattr(m, 'zone_{}'.format('off'), off_fs)
    else: 
        off_fs = Block()
        off_fs.fs = Block()
        off_fs.zone_hours = 0.0 
        off_fs.fs.operating_cost = 0.0
        setattr(m, 'zone_{}'.format('off'), off_fs)

    #Create a surrogate flowsheet for each operating zone
    op_zones = []
    init_flag = 0
    for (i,zone_output) in enumerate(zone_outputs):
        print("Creating instance ", i)
        #zone_output = zone_outputs[zone]
        op_fs = create_model(
            heat_recovery=heat_recovery,
            capital_fs=False,
            calc_boiler_eff=calc_boiler_eff)
        # Set model inputs for the capex and opex plant
        op_fs = set_inputs(op_fs)

        # Fix the p_max of op_fs to p of cap_fs for initialization
        op_fs.fs.net_power_max.fix(value(m.cap_fs.fs.net_cycle_power_output))
        
        #initialize
        if init_flag == 0:
            # Initialize the opex plant
            op_fs = initialize_model(op_fs)

            # save model state after initializing the first instance
            to_json(op_fs.fs, fname="initialized_state.json.gz",
                    gz=True, human_read=True)
            init_flag = 1
        else:
            # Initialize the capex and opex plant
            from_json(op_fs.fs, fname="initialized_state.json.gz", gz=True)

        # Closing the loop in the flowsheet
        op_fs = close_flowsheet_loop(op_fs)
        op_fs = add_operating_cost(op_fs)

        # Unfix op_fs p_max and set constraint linking that to cap_fs p_max
        op_fs.fs.net_power_max.unfix()
        op_fs.fs.eq_p_max = Constraint(
            expr=op_fs.fs.net_power_max ==
            m.cap_fs.fs.net_cycle_power_output*1e-6
        )

        #Fix zone power output
        op_fs.fs.eq_fix_power = Constraint(expr=op_fs.fs.net_cycle_power_output*1e-6 == zone_output*(m.pmax-m.pmin) + m.pmin)
        
        #NOTE: we have to use setattr and getattr since putting optml blocks on sublocks doesn't work
        op_fs.zone_hours_surrogate = Var()
        setattr(m, 'nn_{}'.format(i+1), optml.OptMLBlock())

        #build the formulation on the OptML block
        formulation = optml.neuralnet.ReducedSpaceContinuousFormulation(zone_defns[i+1])
        getattr(m,"nn_{}".format(i+1)).build_formulation(formulation,input_vars=m.inputs, output_vars=[op_fs.zone_hours_surrogate])

        #smooth max on zone hours (avoids negative hours)
        op_fs.zone_hours = Expression(expr=0.5*pyo.sqrt(op_fs.zone_hours_surrogate**2 + 0.001**2) + 0.5*op_fs.zone_hours_surrogate)

        #unfix the boiler flow rate
        op_fs.fs.boiler.inlet.flow_mol[0].setlb(0.01)
        op_fs.fs.boiler.inlet.flow_mol[0].unfix()
        setattr(m, 'zone_{}'.format(i), op_fs)
        op_zones.append(op_fs)

    #scale hours between 0 and 1 year (8736 hours)
    m.zone_total_hours = sum(op_zones[i].zone_hours for i in range(len(op_zones))) + off_fs.zone_hours
    for op_fs in op_zones:
        op_fs.scaled_zone_hours = Var(within=NonNegativeReals, bounds=(0,8736), initialize=100)
        #scaled_hours_i = surrogate_i * 8736 / surrogate_total
        op_fs.con_scale_zone_hours = Constraint(expr = op_fs.scaled_zone_hours*m.zone_total_hours == op_fs.zone_hours*8736)

    if include_zone_off:
        off_fs.scaled_zone_hours = Var(within=NonNegativeReals, bounds=(0,8736), initialize=100)
        off_fs.con_scale_zone_hours = Constraint(expr = off_fs.scaled_zone_hours*m.zone_total_hours == off_fs.zone_hours*8736)
    else:
        off_fs.scaled_zone_hours = 0.0
        
    #operating cost in $MM (million dollars)
    m.op_expr = sum(op_zones[i].scaled_zone_hours*op_zones[i].fs.operating_cost for i in range(len(op_zones)))*1e-6 + \
    off_fs.scaled_zone_hours*off_fs.fs.operating_cost*1e-6

    m.op_zones = op_zones

    #Piecewise cost limits, connect marginal cost to operating cost
    m.cost_lower = Constraint(expr = m.pmin*m.marg_cst <= op_zones[0].fs.operating_cost)   #cost at pmin
    m.cost_upper = Constraint(expr = m.pmax*m.marg_cst >= op_zones[-1].fs.operating_cost)  #cost at pmax

    # Expression for total cap and op cost - $
    #m.total_cost = Expression(expr=plant_lifetime*m.op_expr + capital_payment_years*cap_expr)
    m.total_cost = Expression(expr=plant_lifetime*m.op_expr + capital_payment_years*cap_expr)

    # Expression for total revenue
    m.total_revenue = Expression(expr=plant_lifetime*m.revenue)

    # Objective $
    m.obj = Objective(expr=-(m.total_revenue - m.total_cost))
    #m.obj = Objective(expr=-(m.total_revenue))

    # Unfixing the boiler inlet flowrate for capex plant
    m.cap_fs.fs.boiler.inlet.flow_mol[0].unfix()

    # Setting bounds for the capex plant flowrate
    m.cap_fs.fs.boiler.inlet.flow_mol[0].setlb(0.01)
    # m.cap_fs.fs.boiler.inlet.flow_mol[0].setub(25000)

    # Setting bounds for net cycle power output for the capex plant
    m.cap_fs.fs.eq_min_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output >= p_lower_bound*1e6)

    m.cap_fs.fs.eq_max_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output <= p_upper_bound*1e6)

    return m
