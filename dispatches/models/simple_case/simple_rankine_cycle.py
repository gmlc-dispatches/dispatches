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
    TransformationFactory, value, Block, Expression, Constraint
from pyomo.network import Arc

# Import IDAES components
from idaes.core import FlowsheetBlock, UnitModelBlockData

# Import heat exchanger unit model
from idaes.generic_models.unit_models import Heater, PressureChanger

from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
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

    # expand arcs
    TransformationFactory("network.expand_arcs").apply_to(m)

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

    print('Net power = ', value(m.fs.net_cycle_power_output)*1e-6, ' MW')
    print('Cycle efficiency = ', value(m.fs.cycle_efficiency))


def set_inputs(m):

    # Boiler inlet
    m.fs.boiler.inlet.flow_mol[0].fix(10000)  # mol/s
    m.fs.boiler.inlet.pressure[0].fix(24.23e6)  # MPa
    m.fs.boiler.inlet.enth_mol[0].fix(
        htpx(T=563.6*units.K,
             P=value(m.fs.boiler.inlet.pressure[0])*units.Pa))

    # unit specifications
    m.fs.boiler.outlet.enth_mol[0].fix(
        htpx(T=866.5*units.K,
             P=value(m.fs.boiler.inlet.pressure[0])*units.Pa))

    m.fs.turbine.ratioP.fix(0.05)
    m.fs.turbine.efficiency_isentropic.fix(0.94)

    m.fs.condenser.outlet.pressure[0].fix(6894)  # Pa
    m.fs.condenser.outlet.enth_mol[0].fix(
        htpx(T=311*units.K,
             P=value(m.fs.condenser.outlet.pressure[0])*units.Pa))

    m.fs.bfw_pump.efficiency_pump.fix(0.80)
    m.fs.bfw_pump.deltaP.fix(24.23e6)

    return m


if __name__ == "__main__":

    m = create_model()

    m = set_inputs(m)

    m = initialize_model(m)

    generate_report(m)

    m.fs.boiler.inlet.flow_mol[0].unfix()

    # Net power constraint
    m.fs.eq_net_power = Constraint(
        expr=m.fs.net_cycle_power_output == 100e6
    )

    solver = get_default_solver()
    solver.solve(m, tee=True)

    generate_report(m)
