#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
"""
Simple rankine cycle model. Has couple of options:
1. Recover waste heat after turbine to mimic feed water heater integration
2. Option to include boiler efficiency which is a linear fit f(capacity factor)

if no heat recovery, the flowsheet is as follows:
    Boiler --> Turbine --> Condenser --> Pump --> Boiler

if heat_recovery, the flowsheet is as follows:
    Boiler --> Turbine --> pre-condenser(- Q_recovered) --> Condenser -->
    Pump --> Feed water heater(+ Q_recovered) --> Boiler

Note:
* Boiler and condenser are simple heater blocks
* IAPWS95 for water and steam properties
"""

__author__ = "Jaffer Ghouse"


# Import Pyomo libraries
from pyomo.environ import (
    ConcreteModel, units, Var, TransformationFactory, value, Block, 
    Expression, Constraint, Param, Objective, units as pyunits,
)
from pyomo.network import Arc

# Import IDAES components
from idaes.core import FlowsheetBlock, UnitModelBlockData, UnitModelCostingBlock

# Import heat exchanger unit model
from idaes.models.unit_models import Heater, PressureChanger

from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models_extra.power_generation.costing.power_plant_capcost \
    import QGESSCosting, QGESSCostingData

# Import steam property package
from idaes.models.properties.iapws95 import htpx, Iapws95ParameterBlock

from idaes.apps.grid_integration import MultiPeriodModel
from idaes.core.util import to_json, from_json
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers import get_solver

# Import IDAES logger
import idaes.logger as idaeslog


def create_model(
    m=None,
    heat_recovery=False, 
    calc_boiler_eff=False, 
    capital_fs=False,
):

    if m is None:
        m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.steam_prop = Iapws95ParameterBlock()

    m.fs.boiler = Heater(
        property_package=m.fs.steam_prop,
        has_pressure_change=False,
    )

    m.fs.turbine = PressureChanger(
        property_package=m.fs.steam_prop,
        compressor=False,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    if heat_recovery:
        m.fs.pre_condenser = Heater(
            property_package=m.fs.steam_prop,
            has_pressure_change=True,
        )

        # Spec for pre-condenser
        m.fs.pre_condenser.eq_outlet_cond = Constraint(
            expr=m.fs.pre_condenser.control_volume.
            properties_out[0].enth_mol == m.fs.pre_condenser.control_volume.
            properties_out[0].enth_mol_sat_phase["Liq"]
        )

        m.fs.feed_water_heater = Heater(
            property_package=m.fs.steam_prop,
            has_pressure_change=True,
        )

        # Link precondenser heat and feed water heater
        m.fs.eq_heat_recovery = Constraint(
            expr=m.fs.pre_condenser.heat_duty[0] ==
            - m.fs.feed_water_heater.heat_duty[0]
        )

    m.fs.condenser = Heater(
        property_package=m.fs.steam_prop,
        has_pressure_change=True,
    )

    m.fs.bfw_pump = PressureChanger(
        property_package=m.fs.steam_prop,
        thermodynamic_assumption=ThermodynamicAssumption.pump,
    )

    # create arcs
    m.fs.boiler_to_turbine = Arc(source=m.fs.boiler.outlet,
                                 destination=m.fs.turbine.inlet)

    if heat_recovery:
        m.fs.turbine_to_precondenser = Arc(
            source=m.fs.turbine.outlet,
            destination=m.fs.pre_condenser.inlet)
        m.fs.precondenser_to_condenser = Arc(
            source=m.fs.pre_condenser.outlet,
            destination=m.fs.condenser.inlet)
        m.fs.pump_to_feedwaterheater = Arc(
            source=m.fs.bfw_pump.outlet,
            destination=m.fs.feed_water_heater.inlet)
    else:
        m.fs.turbine_to_condenser = Arc(source=m.fs.turbine.outlet,
                                        destination=m.fs.condenser.inlet)

    m.fs.condenser_to_pump = Arc(source=m.fs.condenser.outlet,
                                 destination=m.fs.bfw_pump.inlet)

    # expand arcs
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Compute gross power
    m.fs.gross_cycle_power_output = Expression(
        expr=(-m.fs.turbine.work_mechanical[0] - m.fs.bfw_pump.work_mechanical[0])
    )

    # account for generator loss = 5% of gross power output
    m.fs.net_cycle_power_output = Expression(
        expr=0.95 * m.fs.gross_cycle_power_output
    )

    if capital_fs or not calc_boiler_eff:
        # if fs is a capital cost fs, then the P is at P_max and hence
        # set boiler efficiency to value at P_max instead of computing
        m.fs.boiler_eff = Param(initialize=0.95)

    if calc_boiler_eff:

        # Var for net_power max variable.
        # This is needed to compute boiler efficiency as a function of
        # capacity factor; p and p_max must be in MWs
        m.fs.net_power_max = Var(initialize=100, units=pyunits.MW)

        # # Boiler efficiency
        # # Linear fit as function of capacity factor; at P_max eff. is 95%
        m.fs.boiler_eff = Expression(
            expr=0.2143 * (pyunits.convert(m.fs.net_cycle_power_output, pyunits.MW) / m.fs.net_power_max)
            + 0.7357
        )

    #  cycle efficiency
    m.fs.cycle_efficiency = Expression(
        expr=m.fs.net_cycle_power_output/m.fs.boiler.heat_duty[0]
        * m.fs.boiler_eff * 100
    )

    m.heat_recovery = heat_recovery
    m.calc_boiler_eff = calc_boiler_eff

    return m


def initialize_model(m, outlvl=idaeslog.INFO):

    # Deactivate the constraint linking the pre_condenser Q and feed water Q
    if m.heat_recovery:
        m.fs.eq_heat_recovery.deactivate()

    # Check for degrees of freedom before proceeding with initialization
    assert degrees_of_freedom(m) == 0

    # Proceed with initialization
    m.fs.boiler.initialize(outlvl=outlvl)
    propagate_state(m.fs.boiler_to_turbine)

    m.fs.turbine.initialize(outlvl=outlvl)

    if m.heat_recovery:
        propagate_state(m.fs.turbine_to_precondenser)
        m.fs.pre_condenser.initialize(outlvl=outlvl)

        propagate_state(m.fs.precondenser_to_condenser)
        m.fs.condenser.initialize()

        propagate_state(m.fs.condenser_to_pump)
        m.fs.bfw_pump.initialize(outlvl=outlvl)

        propagate_state(m.fs.pump_to_feedwaterheater)
        m.fs.feed_water_heater.initialize(outlvl=outlvl)
    else:
        propagate_state(m.fs.turbine_to_condenser)
        m.fs.condenser.initialize(outlvl=outlvl)

        propagate_state(m.fs.condenser_to_pump)
        m.fs.bfw_pump.initialize(outlvl=outlvl)

    solver = get_solver()
    solver.solve(m, tee=False)

    if m.heat_recovery:
        # Unfix feed water heater temperature as the constraint linking Q
        # will be activated
        m.fs.feed_water_heater.outlet.enth_mol[0].unfix()

        m.fs.eq_heat_recovery.activate()

    assert degrees_of_freedom(m) == 0
    solver.solve(m, tee=True)

    return m


def generate_report(m, unit_model_report=True):

    # Print reports
    if unit_model_report:
        for i in m.fs.component_objects(Block):
            if isinstance(i, UnitModelBlockData):
                i.report()

    print()
    print('Net power = ', value(m.fs.net_cycle_power_output)*1e-6, ' MW')
    print('Cycle efficiency = ', value(m.fs.cycle_efficiency), "%")
    print('Heat rate = ', value(m.fs.heat_rate), 'Btu/kWh')
    print('Boiler feed water flow = ',
          value(m.fs.boiler.inlet.flow_mol[0]), "mol/s")
    print()
    try:
        print('Capital cost = ', value(m.fs.capital_cost), '$M')
    except AttributeError:
        print("No cap cost for opex plant")
    try:
        print('Operating cost =  ',
              value(m.fs.operating_cost/(m.fs.net_cycle_power_output*1e-6)),
              '$/MWh')
    except AttributeError:
        print("No operating cost for capex plant")


def set_inputs(m, bfw_pressure=24.23e6, bfw_flow=10000):
    # Boiler inlet
    m.fs.boiler.inlet.flow_mol[0].fix(bfw_flow)  # mol/s
    m.fs.boiler.inlet.pressure[0].fix(bfw_pressure)  # MPa
    m.fs.boiler.inlet.enth_mol[0].fix(
        htpx(T=563.6*units.K, P=value(m.fs.boiler.inlet.pressure[0])*units.Pa)
    )

    # Unit specifications
    m.fs.boiler.outlet.enth_mol[0].fix(
        htpx(T=866.5*units.K, P=value(m.fs.boiler.inlet.pressure[0])*units.Pa)
    )

    turbine_pressure_ratio = 2e6 / bfw_pressure
    m.fs.turbine.ratioP.fix(turbine_pressure_ratio)
    m.fs.turbine.efficiency_isentropic.fix(0.85)

    if m.heat_recovery:
        # precondenser
        m.fs.pre_condenser.deltaP.fix(-0.5e6)  # Pa

        # feed water heater
        m.fs.feed_water_heater.deltaP[0].fix(0)  # Pa
        m.fs.feed_water_heater.outlet.enth_mol[0].fix(
            htpx(T=563.6*units.K, P=value(m.fs.condenser.outlet.pressure[0])*units.Pa)
        )

    m.fs.condenser.outlet.pressure[0].fix(1.05e6)  # Pa
    m.fs.condenser.outlet.enth_mol[0].fix(
        htpx(T=311*units.K, P=value(m.fs.condenser.outlet.pressure[0])*units.Pa)
    )

    m.fs.bfw_pump.efficiency_pump.fix(0.80)
    m.fs.bfw_pump.deltaP.fix(bfw_pressure)

    return m


def close_flowsheet_loop(m):

    """Closes the loop i.e. the arc between the feed water heater and
    boiler. When the pressure and enthalpy arcs are enabled, the bfw_pump
    spec for deltaP and the inlet enth_mol for the boiler need to be unfixed.

    Returns:
        m: model object after closing the loop
    """
    # Unfix bfw pump pressure spec
    m.fs.bfw_pump.deltaP.unfix()

    # Unfix inlet boiler enthalpy
    m.fs.boiler.inlet.enth_mol[0].unfix()

    if m.heat_recovery:
        # Constraint to link pressure
        m.fs.eq_pressure = Constraint(
            expr=m.fs.feed_water_heater.outlet.pressure[0] ==
            m.fs.boiler.inlet.pressure[0]
        )

        # Constraint to link enthalpy
        m.fs.eq_enthalpy = Constraint(
            expr=m.fs.feed_water_heater.outlet.enth_mol[0] ==
            m.fs.boiler.inlet.enth_mol[0]
        )
    else:
        # Constraint to link pressure
        m.fs.eq_pressure = Constraint(
            expr=m.fs.bfw_pump.outlet.pressure[0] ==
            m.fs.boiler.inlet.pressure[0]
        )

        # Constraint to link enthalpy
        m.fs.eq_enthalpy = Constraint(
            expr=m.fs.bfw_pump.outlet.enth_mol[0] ==
            m.fs.boiler.inlet.enth_mol[0]
        )

    return m


def add_capital_cost(m):

    """Add capital cost expressions. Leverages costing correlations from the
    IDAES costing library. Note that the capital cost correlations are all
    based on the boiler feed water flowrate.

    Returns:
        m: model object after adding capital cost correlations
    """
    m.fs.costing = QGESSCosting()

    # Add boiler capital cost
    boiler_power_account = ['4.9']
    # convert flow rate of BFW from mol/s to lb/hr for costing expressions
    m.fs.bfw_lb_hr = Expression(
        expr=pyunits.convert(
            m.fs.boiler.inlet.flow_mol[0] * m.fs.steam_prop.mw, pyunits.lb / pyunits.hr
        )
    )

    m.fs.boiler.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": boiler_power_account,
            "scaled_param": m.fs.bfw_lb_hr,
            "tech": 2,
        },
    )

    # Add turbine capital cost
    turb_power_account = ['8.1']
    # convert the turbine power from W to kW for costing expressions
    m.fs.turbine_power_mw = Expression(
        expr=pyunits.convert(-m.fs.turbine.work_mechanical[0], pyunits.kW)
    )

    m.fs.turbine.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": turb_power_account,
            "scaled_param": m.fs.turbine_power_mw,
            "tech": 2,
        }
    )

    # Add condenser cost
    cond_power_account = ['8.3']
    # convert the heat duty from J/s to MMBtu/hr for costing expressions
    m.fs.condenser_duty_mmbtu_h = Expression(
        expr=pyunits.convert(-m.fs.condenser.heat_duty[0], pyunits.MBtu / pyunits.hr),
    )

    m.fs.condenser.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": cond_power_account,
            "scaled_param": m.fs.condenser_duty_mmbtu_h,
            "tech": 2,
        }
    )

    # Add feed water system costs
    # Note that though no feed water heaters were used, BFW flowrate is used
    # to cost the fed water system
    fwh_power_account = ['3.1', '3.3', '3.5']

    m.fs.bfw_pump.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=QGESSCostingData.get_PP_costing,
        costing_method_arguments={
            "cost_accounts": fwh_power_account,
            "scaled_param": m.fs.bfw_lb_hr,
            "tech": 2,
        }
    )

    # Add expression for total capital cost
    m.fs.capital_cost = Expression(
        expr=m.fs.boiler.costing.total_plant_cost['4.9'] +
        m.fs.turbine.costing.total_plant_cost['8.1'] +
        m.fs.condenser.costing.total_plant_cost['8.3'] +
        sum(m.fs.bfw_pump.costing.total_plant_cost[:]),
        doc="Total capital cost $ Million")

    return m


def add_operating_cost(m, include_cooling_cost=True, coal_price=51.96):

    """Add operating cost expressions. The operating cost only includes
    the cost of coal. This is computed by calculating the amount of coal
    required based on HHV value of coal and the boiler heat duty.

    Returns:
        m: model object after adding operating cost correlations
    """

    # Add condenser cooling water cost
    # temperature for the cooling water from/to cooling tower in K
    t_cw_in = 289.15
    t_cw_out = 300.15

    # compute the delta_h based on fixed temperature of cooling water
    # utility
    m.fs.enth_cw_in = Param(
        initialize=htpx(T=t_cw_in*units.K, P=101325*units.Pa),
        doc="inlet enthalpy of cooling water to condenser",
        units=pyunits.J / pyunits.mol,
    )
    m.fs.enth_cw_out = Param(
        initialize=htpx(T=t_cw_out*units.K, P=101325*units.Pa),
        doc="outlet enthalpy of cooling water from condenser",
        units=pyunits.J / pyunits.mol,
    )
    m.fs.density_cw = Param(
        initialize=1000,
        doc="Density of cooling water in kg/m^3",
        units=pyunits.kg / pyunits.m ** 3,
    )

    m.fs.cw_flow = Expression(
        expr=pyunits.convert(
            -m.fs.condenser.heat_duty[0] /
            (m.fs.density_cw * (m.fs.enth_cw_out - m.fs.enth_cw_in) / m.fs.steam_prop.mw),
            pyunits.gal / pyunits.hr
        ),
        doc="cooling water flow rate in gallons/hr",
    )

    # cooling water cost is $0.19 per 1000 gallons
    m.fs.cw_cost = Param(
        initialize=0.19 / 1000,
        doc="cost of cooling water for condenser in $/gallon",
        units=pyunits.USD_2018 / pyunits.gal,
    )

    m.fs.cw_total_cost = Expression(
        expr=m.fs.cw_flow * m.fs.cw_cost,
        doc="total cooling water cost in $/hr",
    )

    # Add coal feed costs
    # HHV value of coal (Reference - NETL baseline report rev #4)
    m.fs.coal_hhv = Param(
        initialize=27113,
        doc="Higher heating value of coal as received kJ/kg",
        units=pyunits.kJ / pyunits.kg,
    )

    # cost of coal (Reference - NETL baseline report rev #4)
    m.fs.coal_cost = Param(
        initialize=coal_price,
        doc="$ per ton of Illinois no. 6 coal",
        units=pyunits.USD_2018 / pyunits.ton,
    )

    # Expression to compute coal flow rate in ton/hr using Q_boiler and
    # hhv values
    m.fs.coal_flow = Expression(
        expr=pyunits.convert(
            (m.fs.boiler.heat_duty[0] / m.fs.boiler_eff) 
            / pyunits.convert(m.fs.coal_hhv, pyunits.J / pyunits.kg),
            pyunits.ton / pyunits.hr
        ),
        doc="coal flow rate for boiler ton/hr",
    )
    # Expression to compute total cost of coal feed in $/hr
    m.fs.total_coal_cost = Expression(
        expr=m.fs.coal_flow * m.fs.coal_cost,
        doc="total cost of coal feed in $/hr"
    )

    # Expression to compute heat rate (Btu/kWh)
    # Factors:
    # 907.18 to convert from ton to Kg
    # 0.9478 to convert 1 KJ to 1 BTU
    # 1e3 to convert power in MW to kW

    m.fs.heat_rate = Expression(
        expr=pyunits.convert(m.fs.coal_flow * m.fs.coal_hhv, pyunits.Btu / pyunits.hr)
        / pyunits.convert(m.fs.net_cycle_power_output, pyunits.kW),
        doc="heat rate of plant in Btu/kWh",
    )

    if include_cooling_cost:
        # Expression for total operating cost
        m.fs.operating_cost = Expression(
            expr=m.fs.total_coal_cost + m.fs.cw_total_cost,
            doc="Total operating cost in $/hr",
        )
    else:
        # Expression for total operating cost
        m.fs.operating_cost = Expression(
            expr=m.fs.total_coal_cost,
            doc="Total operating cost in $/hr",
        )

    return m


def square_problem(
    heat_recovery=None,
    capital_fs=False,
    net_power=100,
    p_max=100,
    calc_boiler_eff=False,
    capital_payment_years=5,
):

    """This method simulates the simple rankine cycle by adding capital and
    operating costs.

    """
    # Create plant flowsheet
    m = create_model(
        heat_recovery=heat_recovery,
        capital_fs=capital_fs,
        calc_boiler_eff=calc_boiler_eff)

    # Set model inputs for the capex and opex plant
    set_inputs(m)

    # Set p_max for plant that is set in a square problem
    if calc_boiler_eff:
        m.fs.net_power_max.fix(p_max)

    # Initialize the capex and opex plant
    initialize_model(m)

    # Closing the loop in the flowsheet
    close_flowsheet_loop(m)

    # Unfixing the boiler inlet flowrate
    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Net power constraint for the capex plant
    m.fs.eq_net_power = Constraint(
        expr=m.fs.net_cycle_power_output == net_power*1e6
    )

    add_capital_cost(m)

    add_operating_cost(m, include_cooling_cost=True)

    # Expression for total cap and op cost - $/hr
    m.total_cost = Expression(
        expr=(m.fs.capital_cost*1e6/capital_payment_years/8760) +
        m.fs.operating_cost)

    solver = get_solver()
    solver.solve(m, tee=True)

    generate_report(m, unit_model_report=False)

    return m


def stochastic_optimization_problem(
    heat_recovery=False,
    calc_boiler_eff=False,
    p_max_lower_bound=10,
    p_max_upper_bound=300,
    capital_payment_years=5,
    plant_lifetime=20,
    power_demand=None, 
    lmp=None,
    lmp_weights=None,
):

    """This method sets up the stochastic optimization problem that sets up a
    steady-state, pricetaker problem.

    Returns:
        [type]: [description]
    """
    m = ConcreteModel()

    # Create capex plant
    m.cap_fs = create_model(
        heat_recovery=heat_recovery,
        capital_fs=True, 
        calc_boiler_eff=False,
    )
    set_inputs(m.cap_fs)
    initialize_model(m.cap_fs)
    close_flowsheet_loop(m.cap_fs)
    add_capital_cost(m.cap_fs)

    # capital cost (M$/yr)
    cap_expr = pyunits.convert(
        m.cap_fs.fs.capital_cost / capital_payment_years, pyunits.USD_2018
    )

    # Create opex plant
    op_expr = 0
    rev_expr = 0

    for i in range(len(lmp)):

        print()
        print("Creating instance ", i)
        if not calc_boiler_eff:
            op_fs = create_model(
                heat_recovery=heat_recovery,
                capital_fs=False,
                calc_boiler_eff=False)

            # Set model inputs for the capex and opex plant
            set_inputs(op_fs)

            if i == 0:
                # Initialize the capex and opex plant
                initialize_model(op_fs)

                # save model state after initializing the first instance
                to_json(op_fs.fs, fname="initialized_state.json.gz",
                        gz=True, human_read=True)
            else:
                # Initialize the capex and opex plant
                from_json(op_fs.fs, fname="initialized_state.json.gz", gz=True)

            # Closing the loop in the flowsheet
            close_flowsheet_loop(op_fs)

            add_operating_cost(op_fs)

            op_expr += lmp_weights[i]*op_fs.fs.operating_cost
            rev_expr += lmp_weights[i]*lmp[i]*op_fs.fs.net_cycle_power_output*1e-6

            # Add inequality constraint linking net power to cap_ex
            # operating P_min <= 30% of design P_max
            op_fs.fs.eq_min_power = Constraint(
                expr=op_fs.fs.net_cycle_power_output >=
                0.3*m.cap_fs.fs.net_cycle_power_output)
            # operating P_max = design P_max
            op_fs.fs.eq_max_power = Constraint(
                expr=op_fs.fs.net_cycle_power_output <=
                m.cap_fs.fs.net_cycle_power_output)
        else:
            op_fs = create_model(
                heat_recovery=heat_recovery,
                capital_fs=False,
                calc_boiler_eff=True)

            # Set model inputs for the capex and opex plant
            set_inputs(op_fs)

            # Fix the p_max of op_fs to p of cap_fs for initialization
            op_fs.fs.net_power_max.fix(
                value(m.cap_fs.fs.net_cycle_power_output))

            if i == 0:
                # Initialize the capex and opex plant
                initialize_model(op_fs)

                # save model state after initializing the first instance
                to_json(op_fs.fs, fname="initialized_state.json.gz",
                        gz=True, human_read=True)
            else:
                # Initialize the capex and opex plant
                from_json(op_fs.fs, fname="initialized_state.json.gz", gz=True)

            # Closing the loop in the flowsheet
            close_flowsheet_loop(op_fs)
            add_operating_cost(op_fs)

            op_expr += lmp_weights[i]*op_fs.fs.operating_cost
            rev_expr += lmp_weights[i]*lmp[i]*op_fs.\
                fs.net_cycle_power_output*1e-6

            # Unfix op_fs p_max and set constraint linking that to cap_fs p_max
            op_fs.fs.net_power_max.unfix()
            op_fs.fs.eq_p_max = Constraint(
                expr=op_fs.fs.net_power_max ==
                m.cap_fs.fs.net_cycle_power_output*1e-6
            )

            # Add inequality constraint linking net power to cap_ex
            # operating P_min <= 30% of design P_max
            op_fs.fs.eq_min_power = Constraint(
                expr=op_fs.fs.net_cycle_power_output >=
                0.3*m.cap_fs.fs.net_cycle_power_output)
            # operating P_max = design P_max
            op_fs.fs.eq_max_power = Constraint(
                expr=op_fs.fs.net_cycle_power_output <=
                m.cap_fs.fs.net_cycle_power_output)

        # only if power demand is given
        if power_demand is not None:
            op_fs.fs.eq_max_produced = Constraint(
                expr=op_fs.fs.net_cycle_power_output <=
                power_demand[i]*1e6)

        op_fs.fs.boiler.inlet.flow_mol[0].unfix()

        # Set bounds for the flow
        op_fs.fs.boiler.inlet.flow_mol[0].setlb(1)
        # op_fs.fs.boiler.inlet.flow_mol[0].setub(25000)

        setattr(m, 'scenario_{}'.format(i), op_fs)

    # Expression for total cap and op cost - $
    m.total_cost = Expression(
        expr=plant_lifetime*op_expr + capital_payment_years*cap_expr)

    # Expression for total revenue
    m.total_revenue = Expression(
        expr=plant_lifetime*rev_expr)

    # Objective $
    m.obj = Objective(
        expr=-(m.total_revenue - m.total_cost))

    # Unfixing the boiler inlet flowrate for capex plant
    m.cap_fs.fs.boiler.inlet.flow_mol[0].unfix()

    # Setting bounds for the capex plant flowrate
    m.cap_fs.fs.boiler.inlet.flow_mol[0].setlb(5)

    # Setting bounds for net cycle power output for the capex plant
    m.cap_fs.fs.eq_min_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output >= p_max_lower_bound*1e6)

    m.cap_fs.fs.eq_max_power = Constraint(
        expr=m.cap_fs.fs.net_cycle_power_output <=
        p_max_upper_bound*1e6)

    return m


def unfix_dof_function(m, cap_fs):
    
    # Unfix inlet flowrate to boiler
    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Set bounds for the flow
    m.fs.boiler.inlet.flow_mol[0].setlb(1)

    if m.calc_boiler_eff:
        m.fs.net_power_max.unfix()

        m.fs.eq_p_max = Constraint(
            expr=m.fs.net_power_max == cap_fs.fs.net_cycle_power_output * 1e-6
        )

    close_flowsheet_loop(m)


if __name__ == "__main__":

    # Code to generate op cost, heat rate, eff vs. capacity factor plot
    """
    p_max = 300
    p_min = 90
    power = list(reversed(range(p_min, p_max + 30, 30)))
    plant_capacity = [p*100/p_max for p in power]
    cycle_eff = []
    heat_rate = []
    op_cost = []
    for i in power:
        print(i)
        m = square_problem(
            heat_recovery=True,
            capital_fs=False,
            calc_boiler_eff=True,
            p_max=p_max, net_power=i)
        cycle_eff.append(value(m.fs.cycle_efficiency))
        heat_rate.append(value(m.fs.heat_rate))
        op_cost.append(value(m.fs.operating_cost)/i)

    # plots
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    plt.plot(plant_capacity, op_cost, color="green")
    sec_yaxis = ax.secondary_yaxis('left')

    ax.set_xlabel("operating capacity (%)")
    ax.set_ylabel("operating cost ($/MWh)", color="green")

    ax1 = ax.twinx()
    ax1.plot(plant_capacity, heat_rate, color="green")
    ax1.yaxis.set_ticks_position('left')
    ax1.yaxis.set_label_position('left')
    ax1.spines['left'].set_position(('outward', 75))
    ax1.set_ylabel("heat rate (BTU/kWh)", color="green")

    ax2 = ax.twinx()
    ax2.plot(plant_capacity, cycle_eff, color="red")
    ax2.set_ylabel("cycle efficiency (%)", color="red")
    plt.grid()
    # plt.savefig("operating_cost_vs_plant_capacity.pdf",
    #             format="pdf",
    #             bbox_inches="tight")
    plt.show()
    """

    # Code to generate capital cost as f(P_max)
    m = square_problem(
        heat_recovery=True,
        capital_fs=True,
        calc_boiler_eff=True,
        p_max=300, net_power=300)
