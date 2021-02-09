##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Simple rankine cycle model.

Boiler --> Turbine --> Condenser --> Pump --> Boiler

Note:
* Boiler and condenser are simple heater blocks
* IAPWS95 for water and steam properties
"""

__author__ = "Jaffer Ghouse"


# Import Pyomo libraries
from pyomo.environ import ConcreteModel, SolverFactory, units, \
    TransformationFactory, value, Block, Expression, Constraint, Param
from pyomo.network import Arc

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
from idaes.core.util.testing import get_default_solver
import idaes.logger as idaeslog


def create_model():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(default={"dynamic": False})

    m.fs.steam_prop = Iapws95ParameterBlock()

    m.fs.boiler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.steam_prop,
            "has_pressure_change": False})

    m.fs.turbine = PressureChanger(
        default={
            "property_package": m.fs.steam_prop,
            "compressor": False,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic})

    m.fs.condenser = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.steam_prop,
            "has_pressure_change": True})

    m.fs.bfw_pump = PressureChanger(
        default={
            "property_package": m.fs.steam_prop,
            "thermodynamic_assumption": ThermodynamicAssumption.pump})

    # create arcs
    m.fs.boiler_to_turbine = Arc(source=m.fs.boiler.outlet,
                                 destination=m.fs.turbine.inlet)

    m.fs.turbine_to_condenser = Arc(source=m.fs.turbine.outlet,
                                    destination=m.fs.condenser.inlet)

    m.fs.condenser_to_pump = Arc(source=m.fs.condenser.outlet,
                                 destination=m.fs.bfw_pump.inlet)

    # m.fs.pump_to_boiler = Arc(source=m.fs.bfw_pump.outlet,
    #                           destination=m.fs.boiler.inlet)
    # expand arcs
    TransformationFactory("network.expand_arcs").apply_to(m)
    # deactivate the flow equality
    m.fs.gross_cycle_power_output = \
        Expression(expr=(-m.fs.turbine.work_mechanical[0] -
                   m.fs.bfw_pump.work_mechanical[0]))

    # account for generator loss = 1.5% of gross power output
    m.fs.net_cycle_power_output = Expression(
        expr=0.985*m.fs.gross_cycle_power_output)

    #  cycle efficiency
    m.fs.cycle_efficiency = Expression(
        expr=m.fs.gross_cycle_power_output/m.fs.boiler.heat_duty[0] * 100
    )

    return m


def initialize_model(m, outlvl=idaeslog.INFO):

    assert degrees_of_freedom(m) == 0

    # proceed with initialization
    m.fs.boiler.initialize(outlvl=outlvl)

    propagate_state(m.fs.boiler_to_turbine)

    m.fs.turbine.initialize(outlvl=outlvl)

    propagate_state(m.fs.turbine_to_condenser)

    m.fs.condenser.initialize(outlvl=outlvl)

    propagate_state(m.fs.condenser_to_pump)

    m.fs.bfw_pump.initialize(outlvl=outlvl)

    solver = get_default_solver()
    solver.solve(m, tee=True)

    return m


def generate_report(m):

    # Print reports
    for i in m.fs.component_objects(Block):
        if isinstance(i, UnitModelBlockData):
            i.report()

    print()
    print('Net power = ', value(m.fs.net_cycle_power_output)*1e-6, ' MW')
    print('Cycle efficiency = ', value(m.fs.cycle_efficiency))

    print()
    print('Capital cost = ', value(m.fs.capital_cost), '$M')
    print('Operating cost =  ', value(m.fs.operating_cost), '$/hr')


def set_inputs(m):

    # Main steam pressure
    bfw_pressure = 24.23e6  # Pa

    # Boiler inlet
    m.fs.boiler.inlet.flow_mol[0].fix(10000)  # mol/s
    m.fs.boiler.inlet.pressure[0].fix(bfw_pressure)  # MPa
    m.fs.boiler.inlet.enth_mol[0].fix(
        htpx(T=563.6*units.K,
             P=value(m.fs.boiler.inlet.pressure[0])*units.Pa))

    # unit specifications
    m.fs.boiler.outlet.enth_mol[0].fix(
        htpx(T=866.5*units.K,
             P=value(m.fs.boiler.inlet.pressure[0])*units.Pa))

    turbine_pressure_ratio = 0.52e6/bfw_pressure
    m.fs.turbine.ratioP.fix(turbine_pressure_ratio)
    m.fs.turbine.efficiency_isentropic.fix(0.94)

    m.fs.condenser.outlet.pressure[0].fix(101325)  # Pa
    m.fs.condenser.outlet.enth_mol[0].fix(
        htpx(T=311*units.K,
             P=value(m.fs.condenser.outlet.pressure[0])*units.Pa))

    m.fs.bfw_pump.efficiency_pump.fix(0.80)
    m.fs.bfw_pump.deltaP.fix(bfw_pressure)

    return m


def close_flowsheet_loop(m):

    # Unfix inlet boiler pressure
    m.fs.boiler.inlet.pressure[0].unfix()

    # Constraint to link pressure
    m.fs.eq_pressure = Constraint(
        expr=m.fs.bfw_pump.outlet.pressure[0] == m.fs.boiler.inlet.pressure[0]
    )

    # Unfix inlet boiler enthalpy
    m.fs.boiler.inlet.enth_mol[0].unfix()

    # Constraint to link enthalpy
    m.fs.eq_enthalpy = Constraint(
        expr=m.fs.bfw_pump.outlet.enth_mol[0] == m.fs.boiler.inlet.enth_mol[0]
    )

    return m


def add_capital_cost(m):

    m.fs.get_costing(year='2018')

    # Add boiler capital cost
    boiler_power_account = ['4.9']
    # convert flow rate of BFW from mol/s to lb/hr for costing expressions
    m.fs.bfw_lb_hr = Expression(
        expr=m.fs.boiler.inlet.flow_mol[0]*0.018*2.204*3600)
    get_PP_costing(
        m.fs.boiler, boiler_power_account, m.fs.bfw_lb_hr, 'lb/hr', 2)

    # Add turbine capital cost
    turb_power_account = ['8.1']
    # convert the turbine power from W to kW for costing expressions
    m.fs.turbine_power_mw = Expression(
        expr=-m.fs.turbine.work_mechanical[0] * 1e-3)
    get_PP_costing(
        m.fs.turbine, turb_power_account,
        m.fs.turbine_power_mw, 'kW', 2)

    # Add condenser cost
    cond_power_account = ['8.3']
    # convert the heat duty from J/s to MMBtu/hr for costing expressions
    m.fs.condenser_duty_mmbtu_h = Expression(
        expr=-m.fs.condenser.heat_duty[0] * 3.412*1e-6)
    get_PP_costing(
        m.fs.condenser, cond_power_account,
        m.fs.condenser_duty_mmbtu_h, "MMBtu/hr", 2)

    # Add feed water system costs
    # Note that though no feed water heaters were used, BFW flowrate is used
    # to cost the fed water system
    fwh_power_account = ['3.1', '3.3', '3.5']
    get_PP_costing(m.fs.bfw_pump, fwh_power_account,
                   m.fs.bfw_lb_hr, 'lb/hr', 2)

    # Add expression for total capital cost
    m.fs.capital_cost = Expression(
        expr=m.fs.boiler.costing.total_plant_cost['4.9'] +
        m.fs.turbine.costing.total_plant_cost['8.1'] +
        m.fs.condenser.costing.total_plant_cost['8.3'] +
        sum(m.fs.bfw_pump.costing.total_plant_cost[:]),
        doc="Total capital cost $ Million")

    return m


def add_operating_cost(m):

    # Add condenser cooling water cost
    # temperature for the cooling water from/to cooling tower in K
    t_cw_in = 289.15
    t_cw_out = 300.15

    # compute the delta_h based on fixed temperature of cooling water
    # utility
    m.fs.enth_cw_in = Param(
        initialize=htpx(T=t_cw_in*units.K, P=101325*units.Pa),
        doc="inlet enthalpy of cooling water to condenser")
    m.fs.enth_cw_out = Param(
        initialize=htpx(T=t_cw_out*units.K, P=101325*units.Pa),
        doc="outlet enthalpy of cooling water from condenser")

    m.fs.cw_flow = Expression(
        expr=-m.fs.condenser.heat_duty[0]*0.018*0.26417*3600 /
        (m.fs.enth_cw_out-m.fs.enth_cw_in),
        doc="cooling water flow rate in gallons/hr")

    # cooling water cost in $/1000 gallons
    m.fs.cw_cost = Param(
        initialize=1.9,
        doc="cost of cooling water for condenser in $/1000 gallon")   

    m.fs.cw_total_cost = Expression(
        expr=m.fs.cw_flow*m.fs.cw_cost/1000,
        doc="total cooling water cost in $/hr"
    )

    # Add coal feed costs
    # HHV value of coal (Reference - NETL baseline report rev #4)
    m.fs.coal_hhv = Param(
        initialize=27113,
        doc="Higher heating value of coal as received kJ/kg")

    # cost of coal (Reference - NETL baseline report rev #4)
    m.fs.coal_cost = Param(
        initialize=51.96,
        doc="$ per ton of Illinois no. 6 coal"
    )
    # Expression to compute coal flow rate in ton/hr using Q_boiler and
    # hhv values
    m.fs.coal_flow = Expression(
        expr=((m.fs.boiler.heat_duty[0] * 3600)/(907.18*1000*m.fs.coal_hhv)),
        doc="coal flow rate for boiler ton/hr")
    # Expression to compute total cost of coal feed in $/hr
    m.fs.total_coal_cost = Expression(
        expr=m.fs.coal_flow*m.fs.coal_cost,
        doc="total cost of coal feed in $/hr"
    )

    # Expression for total operating cost
    m.fs.operating_cost = Expression(
        expr=m.fs.total_coal_cost+m.fs.cw_total_cost,
        doc="Total operating cost in $/hr")

    return m


def add_npv(m):
    pass


if __name__ == "__main__":

    m = create_model()

    m = set_inputs(m)

    m = initialize_model(m)

    m = close_flowsheet_loop(m)

    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Net power constraint
    m.fs.eq_net_power = Constraint(
        expr=m.fs.net_cycle_power_output == 100e6
    )

    m = add_capital_cost(m)

    m = add_operating_cost(m)

    solver = get_default_solver()
    solver.solve(m, tee=True)

    generate_report(m)
