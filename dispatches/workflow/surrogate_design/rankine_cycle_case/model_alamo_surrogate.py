#the rankine cycle is a directory above this one, so modify path
from pyomo.common.fileutils import this_file_dir
import sys, os, json
sys.path.append(os.path.join(this_file_dir(),"../../../models/simple_case"))

from simple_rankine_cycle import *

#lots of idaes imports
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


#use idaes SurrogateBlock
from idaes.surrogate.alamopy import AlamoSurrogate
from idaes.surrogate.surrogate_block import SurrogateBlock

surrogate_dir = os.path.join(this_file_dir(),"../../train_market_surrogates/steady_state/surrogate_models/alamo/models")

# load scaling and bounds for each surrogate
with open(os.path.join(surrogate_dir,"alamo_parameters_revenue.json"), 'rb') as f:
    rev_data = json.load(f)

with open(os.path.join(surrogate_dir,"alamo_parameters_zones.json"), 'rb') as f:
    zone_data = json.load(f)

with open(os.path.join(surrogate_dir,"alamo_parameters_nstartups.json"), 'rb') as f:
    nstartups_data = json.load(f)


xm = rev_data['xm_inputs']
xstd = rev_data['xstd_inputs']
zm_rev = rev_data['zm_revenue']
zstd_rev = rev_data['zstd_revenue']

zm_zones = zone_data['zm_zones']
zstd_zones = zone_data['zstd_zones']

zm_nstartups = nstartups_data['zm_nstartups']
zstd_nstartups = nstartups_data['zstd_nstartups']


# load surrogates from alamo .json files
alamo_revenue = AlamoSurrogate.load_from_file(os.path.join(surrogate_dir,'alamo_revenue.json'))
alamo_nstartups = AlamoSurrogate.load_from_file(os.path.join(surrogate_dir,'alamo_nstartups.json'))
alamo_zones = AlamoSurrogate.load_from_file(os.path.join(surrogate_dir,'alamo_zones.json'))


#Denote the scaled power output for each of the 10 zones (0 corresponds to pmin, 1.0 corresponds to pmax)
zone_outputs = [0.0,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,1.0]

#this creates the alamo surrogate flowsheet of the rankine cycle
def conceptual_design_problem_alamo(
    heat_recovery=False,
    calc_boiler_eff=False,
    p_lower_bound=10,
    p_upper_bound=500,
    capital_payment_years=5,
    plant_lifetime=20,
    coal_price=51.96
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

    ######################################
    #revenue surrogate
    ######################################
    m.rev_surrogate = Var()
    m.alamo_revenue_surrogate = SurrogateBlock()
    m.alamo_revenue_surrogate.build_model(alamo_revenue)
    input_dict = m.alamo_revenue_surrogate.input_vars_as_dict()

    m.pmax_revenue_in = Constraint(expr = input_dict["pmax"] == (m.pmax - xm[0])/xstd[0])
    m.pmin_revenue_in = Constraint(expr = input_dict["pmin_multi"] == (m.pmin_multi - xm[1])/xstd[1])
    m.ramp_revenue_in = Constraint(expr = input_dict["ramp_multi"] == (m.ramp_multi - xm[2])/xstd[2])
    m.min_up_revenue_in = Constraint(expr = input_dict["min_up_time"] == (m.min_up_time - xm[3])/xstd[3])
    m.min_dn_revenue_in = Constraint(expr = input_dict["min_down_multi"] == (m.min_dn_multi - xm[4])/xstd[4])
    m.marg_cst_revenue_in = Constraint(expr = input_dict["marg_cst"] == (m.marg_cst - xm[5])/xstd[5])
    m.no_load_cst_revenue_in = Constraint(expr = input_dict["no_load_cst"] == (m.no_load_cst - xm[6])/xstd[6])
    m.startup_cst_revenue_in = Constraint(expr = input_dict["startup_cst"] == (m.startup_cst - xm[7])/xstd[7])
    m.revenue_out = Constraint(expr = m.rev_surrogate == m.alamo_revenue_surrogate.outputs['revenue']*zstd_rev + zm_rev)

    #this is a smooth-max; it sets negative revenue to zero
    m.revenue = Expression(expr=0.5*pyo.sqrt(m.rev_surrogate**2 + 0.001**2) + 0.5*m.rev_surrogate)

    #######################################
    #nstartups surrogate
    #######################################
    m.nstartups_surrogate = Var()
    m.alamo_nstartups_surrogate = SurrogateBlock()
    m.alamo_nstartups_surrogate.build_model(alamo_nstartups)
    input_dict = m.alamo_nstartups_surrogate.input_vars_as_dict()

    m.pmax_nstartups_in = Constraint(expr = input_dict["pmax"] == (m.pmax - xm[0])/xstd[0])
    m.pmin_nstartups_in = Constraint(expr = input_dict["pmin_multi"] == (m.pmin_multi - xm[1])/xstd[1])
    m.ramp_nstartups_in = Constraint(expr = input_dict["ramp_multi"] == (m.ramp_multi - xm[2])/xstd[2])
    m.min_up_nstartups_in = Constraint(expr = input_dict["min_up_time"] == (m.min_up_time - xm[3])/xstd[3])
    m.min_dn_nstartups_in = Constraint(expr = input_dict["min_down_multi"] == (m.min_dn_multi - xm[4])/xstd[4])
    m.marg_cst_nstartups_in = Constraint(expr = input_dict["marg_cst"] == (m.marg_cst - xm[5])/xstd[5])
    m.no_load_cst_nstartups_in = Constraint(expr = input_dict["no_load_cst"] == (m.no_load_cst - xm[6])/xstd[6])
    m.startup_cst_nstartups_in = Constraint(expr = input_dict["startup_cst"] == (m.startup_cst - xm[7])/xstd[7])
    m.nstartups_out = Constraint(expr = m.nstartups_surrogate == m.alamo_nstartups_surrogate.outputs['nstartups']*zstd_nstartups + zm_nstartups)

    m.nstartups = Expression(expr=0.5*pyo.sqrt(m.nstartups_surrogate**2 + 0.001**2) + 0.5*m.nstartups_surrogate)

    ############################################
    #zone surrogates
    ############################################
    m.alamo_zones_surrogate = SurrogateBlock()
    m.alamo_zones_surrogate.build_model(alamo_zones)
    input_dict = m.alamo_zones_surrogate.input_vars_as_dict()
    output_dict = m.alamo_zones_surrogate.output_vars_as_dict()

    #the zone surrogate has 11 outputs: off + 10 zones from pmin to pmax
    m.zone_hours_surrogate = Var(range(0,11))

    m.pmax_zones_in = Constraint(expr = input_dict["pmax"] == (m.pmax - xm[0])/xstd[0])
    m.pmin_zones_in = Constraint(expr = input_dict["pmin_multi"] == (m.pmin_multi - xm[1])/xstd[1])
    m.ramp_zones_in = Constraint(expr = input_dict["ramp_multi"] == (m.ramp_multi - xm[2])/xstd[2])
    m.min_up_zones_in = Constraint(expr = input_dict["min_up_time"] == (m.min_up_time - xm[3])/xstd[3])
    m.min_dn_zones_in = Constraint(expr = input_dict["min_down_multi"] == (m.min_dn_multi - xm[4])/xstd[4])
    m.marg_cst_zones_in = Constraint(expr = input_dict["marg_cst"] == (m.marg_cst - xm[5])/xstd[5])
    m.no_load_cst_zones_in = Constraint(expr = input_dict["no_load_cst"] == (m.no_load_cst - xm[6])/xstd[6])
    m.startup_cst_zones_in = Constraint(expr = input_dict["startup_cst"] == (m.startup_cst - xm[7])/xstd[7])

    m.zone_hours_out = Constraint(range(0,11))
    for (i,k) in enumerate(output_dict.keys()):
        m.zone_hours_out[i] = m.zone_hours_surrogate[i] == m.alamo_zones_surrogate.outputs[k]*zstd_zones[i] + zm_zones[i]

    ############################################
    #begin flowsheets
    ############################################
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

        #initialize with json. this speeds up model instantiation. it writes a json file \
        #for the first flowsheet which is used to initialize the next flowsheets
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
        op_fs = add_operating_cost(op_fs, coal_price=coal_price)

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

    #scale zone hours such that they add up to 8736 (if the surrogate is good, the unscaled will be pretty close to this)
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
    m.startup_expr = m.startup_cst*m.nstartups*m.pmax*1e-6 #MM$

    #set zone flowsheets to pyomo model
    m.op_zones = op_zones

    #Piecewise cost limits, connect marginal cost to operating cost. We say marginal cost is the average operating cost
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
