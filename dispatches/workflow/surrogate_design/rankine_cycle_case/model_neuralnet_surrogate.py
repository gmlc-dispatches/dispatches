#the rankine cycle is a directory above this one, so modify path
from pyomo.core.fileutils import this_file_dir
import sys, os, json
sys.path.append(os.join(this_file_dir,"../"))

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

# load scaling and bounds for each surrogate
with open("surrogate_models/scikit/models/training_parameters_revenue.json", 'rb') as f:
    rev_data = json.load(f)

with open("surrogate_models/scikit/models/training_parameters_zones.json", 'rb') as f:
    zone_data = json.load(f)

with open("surrogate_models/scikit/models/training_parameters_nstartups.json", 'rb') as f:
    nstartups_data = json.load(f)

# load surrogates
with open('surrogate_models/scikit/models/scikit_revenue.pkl', 'rb') as f:
    nn_revenue = pickle.load(f)

with open('surrogate_models/scikit/models/scikit_zones.pkl', 'rb') as f:
    nn_zones = pickle.load(f)

with open('surrogate_models/scikit/models/scikit_nstartups.pkl', 'rb') as f:
    nn_nstartups = pickle.load(f)


#load scikit models and create OMLT NetworkDefinition objects
#Revenue model definition
input_bounds_rev = list(zip(rev_data['xmin'],rev_data['xmax']))
scaling_object_revenue = omlt.OffsetScaling(offset_inputs=rev_data['xm_inputs'],
                factor_inputs=rev_data['xstd_inputs'],
                offset_outputs=[rev_data['zm_revenue']],
                factor_outputs=[rev_data['zstd_revenue']])
net_rev_defn = load_scikit_mlp(nn_revenue,scaling_object_revenue,input_bounds_rev)

#Zone model definition
input_bounds_zones = list(zip(zone_data['xmin'],zone_data['xmax']))
scaling_object_zones = omlt.OffsetScaling(offset_inputs=zone_data['xm_inputs'],
                factor_inputs=zone_data['xstd_inputs'],
                offset_outputs=zone_data['zm_zones'],
                factor_outputs=zone_data['zstd_zones'])
net_zone_defn = load_scikit_mlp(nn_zones,scaling_object_zones,input_bounds_zones)

#Nstartup model definition
input_bounds_nstartups = list(zip(nstartups_data['xmin'],nstartups_data['xmax']))
scaling_object_nstartups = omlt.OffsetScaling(offset_inputs=nstartups_data['xm_inputs'],
                factor_inputs=nstartups_data['xstd_inputs'],
                offset_outputs=[nstartups_data['zm_nstartups']],
                factor_outputs=[nstartups_data['zstd_nstartups']])
net_nstartups_defn = load_scikit_mlp(nn_nstartups,scaling_object_nstartups,input_bounds_nstartups)


#Denote the scaled power output for each of the 10 zones (0 corresponds to pmin, 1.0 corresponds to pmax)
zone_outputs = [0.0,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.0]

def conceptual_design_problem_nn(
    heat_recovery=False,
    calc_boiler_eff=False,
    p_lower_bound=10,
    p_upper_bound=500,
    capital_payment_years=5,
    plant_lifetime=20
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

    #surrogate market inputs (not technically part of rankine cycle model but are used in market model)
    m.pmax = Expression(expr = 1.0*m.cap_fs.fs.net_cycle_power_output*1e-6)
    m.pmin_multi = Var(within=NonNegativeReals, bounds=(0.15,0.45), initialize=0.3)  
    m.ramp_multi = Var(within=NonNegativeReals, bounds=(0.5,1.0), initialize=0.5)
    m.min_up_time = Var(within=NonNegativeReals, bounds=(1.0,16.0), initialize=4.0)
    m.min_dn_multi = Var(within=NonNegativeReals, bounds=(0.5,2.0), initialize=4.0)
    m.marg_cst =  Var(within=NonNegativeReals, bounds=(5,30), initialize=5)
    m.no_load_cst =  Var(within=NonNegativeReals, bounds=(0,2.5), initialize=1)
    m.startup_cst = Var(within=NonNegativeReals, bounds=(0,136), initialize=1)
    
    #actual generator values
    m.pmin = Expression(expr = m.pmin_multi*m.pmax)
    m.min_dn_time = Expression(expr = m.min_dn_multi*m.min_up_time)
    m.ramp_rate= Expression(expr =  m.ramp_multi*(m.pmax - m.pmin))

    m.inputs = [m.pmax,m.pmin_multi,m.ramp_multi,m.min_up_time,m.min_dn_multi,m.marg_cst,m.no_load_cst,m.startup_cst]

    ######################################
    #revenue surrogate
    ######################################
    m.rev_surrogate = Var()
    m.nn_rev = omlt.OmltBlock()

    #build the formulation on the omlt block
    formulation_rev = omlt.neuralnet.ReducedSpaceContinuousFormulation(net_rev_defn)
    m.nn_rev.build_formulation(formulation_rev,input_vars=m.inputs, output_vars=[m.rev_surrogate])
    m.revenue = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

    ######################################
    #nstartups surrogate
    ######################################
    m.nstartups_surrogate = Var()
    m.nn_nstartups = omlt.OmltBlock()

    #build the formulation on the omlt block
    formulation_nstartups = omlt.neuralnet.ReducedSpaceContinuousFormulation(net_nstartups_defn)
    m.nn_nstartups.build_formulation(formulation_nstartups,input_vars=m.inputs, output_vars=[m.nstartups_surrogate])
    m.nstartups = Expression(expr=0.5*pyo.sqrt(m.nstartups_surrogate**2 + 0.001**2) + 0.5*m.nstartups_surrogate)

    ############################################
    #zone surrogates
    ############################################
    m.nn_zones = omlt.OmltBlock()
    formulation_zones = omlt.neuralnet.ReducedSpaceContinuousFormulation(net_zone_defn)
    m.zone_set = pyo.Set(initialize=range(0,11))
    m.zone_hours_surrogate = pyo.Var(m.zone_set,within=NonNegativeReals)
    m.nn_zones.build_formulation(formulation_zones,input_vars=m.inputs, output_vars=list(m.zone_hours_surrogate.values()))

    #zone off flowsheet
    off_fs = Block()
    off_fs.fs = Block()
    off_fs.fs.operating_cost = m.no_load_cst*m.pmax 
    off_fs.zone_hours = Expression(expr=0.5*pyo.sqrt(m.zone_hours_surrogate[0]**2 + 0.001**2) + 0.5*m.zone_hours_surrogate[0])
    setattr(m, 'zone_{}'.format('off'), off_fs)


    #Create a surrogate flowsheet for each operating zone
    op_zones = []
    init_flag = 0
    for (i,zone_output) in enumerate(zone_outputs):
        print("Creating instance ", i)
        op_fs = create_model(
            heat_recovery=heat_recovery,
            capital_fs=False,
            calc_boiler_eff=calc_boiler_eff)
        # Set model inputs for the capex and opex plant
        op_fs = set_inputs(op_fs)

        # Fix the p_max of op_fs to p of cap_fs for initialization
        op_fs.fs.net_power_max.fix(value(m.cap_fs.fs.net_cycle_power_output))
        
        #initialize with json
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
        op_fs = add_operating_cost(op_fs, coal_price=12.0)

        # Unfix op_fs p_max and set constraint linking that to cap_fs p_max
        op_fs.fs.net_power_max.unfix()
        op_fs.fs.eq_p_max = Constraint(
            expr=op_fs.fs.net_power_max ==
            m.cap_fs.fs.net_cycle_power_output*1e-6
        )

        #Fix zone power output
        op_fs.fs.eq_fix_power = Constraint(expr=op_fs.fs.net_cycle_power_output*1e-6 == zone_output*(m.pmax-m.pmin) + m.pmin)
        
        #smooth max on zone hours (avoids negative hours)
        op_fs.zone_hours = Expression(expr=0.5*pyo.sqrt(m.zone_hours_surrogate[i+1]**2 + 0.001**2) + 0.5*m.zone_hours_surrogate[i+1])

        #unfix the boiler flow rate
        op_fs.fs.boiler.inlet.flow_mol[0].setlb(0.01)
        op_fs.fs.boiler.inlet.flow_mol[0].unfix()
        setattr(m, 'zone_{}'.format(i), op_fs)
        op_zones.append(op_fs)

    #scale hours between 0 and 1 year (8736 hours were used in simulation)
    m.zone_total_hours = sum(op_zones[i].zone_hours for i in range(len(op_zones))) + off_fs.zone_hours
    for op_fs in op_zones:
        op_fs.scaled_zone_hours = Var(within=NonNegativeReals, bounds=(0,8736), initialize=100)
        # NOTE: scaled_hours_i = surrogate_i * 8736 / surrogate_total
        op_fs.con_scale_zone_hours = Constraint(expr = op_fs.scaled_zone_hours*m.zone_total_hours == op_fs.zone_hours*8736)

    
    off_fs.scaled_zone_hours = Var(within=NonNegativeReals, bounds=(0,8736), initialize=100)
    off_fs.con_scale_zone_hours = Constraint(expr = off_fs.scaled_zone_hours*m.zone_total_hours == off_fs.zone_hours*8736)
    
    #operating cost in $MM (million dollars)
    m.op_expr = sum(op_zones[i].scaled_zone_hours*op_zones[i].fs.operating_cost for i in range(len(op_zones)))*1e-6 + \
    off_fs.scaled_zone_hours*off_fs.fs.operating_cost*1e-6

    #startup cost in MM$
    m.startup_expr = m.startup_cst*m.nstartups*m.pmax*1e-6

    m.op_zones = op_zones

    #Piecewise cost limits, connect marginal cost to operating cost
    m.connect_mrg_cost = Constraint(expr = m.marg_cst == 0.5*(op_zones[0].fs.operating_cost/m.pmin + op_zones[-1].fs.operating_cost/m.pmax)) 

    # Expression for total cap and op cost - $
    m.total_cost = Expression(expr=plant_lifetime*(m.op_expr  + m.startup_expr)+ capital_payment_years*cap_expr)

    # Expression for total revenue
    m.total_revenue = Expression(expr=plant_lifetime*m.revenue)

    # Objective $
    m.obj = Objective(expr=-(m.total_revenue - m.total_cost))

    # Unfixing the boiler inlet flowrate for capex plant
    m.cap_fs.fs.boiler.inlet.flow_mol[0].unfix()

    # Setting bounds for the capex plant flowrate
    m.cap_fs.fs.boiler.inlet.flow_mol[0].setlb(0.01)

    # Setting bounds for net cycle power output for the capex plant
    m.cap_fs.fs.eq_min_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output >= p_lower_bound*1e6)

    m.cap_fs.fs.eq_max_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output <= p_upper_bound*1e6)

    return m