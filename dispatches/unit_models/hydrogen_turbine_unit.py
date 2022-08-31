#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
"""
Turbo-Generator Set for a Hydrogen turbine.

Compressor -> Stoichiometric Reactor -> Turbine
Author: Konor Frick
Date: April 2, 2021
Notes: it is noted that in this example the hydrogen is compressed along
with the air in the compressor as opposed to having a separate
fuel injection system. Noting this is a simplified version of the H2 turbine.
"""
from pyomo.environ import Constraint, Var, TransformationFactory
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.network import Arc

from idaes.models.unit_models import Compressor, \
    StoichiometricReactor, Turbine

import idaes.logger as idaeslog

from idaes.core.util.config import is_physical_parameter_block, \
    is_reaction_parameter_block
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
from idaes.core import declare_process_block_class, UnitModelBlockData, \
    useDefault


@declare_process_block_class(
    "HydrogenTurbine",
    doc="Simple 0D hydrogen turbine unit model")
class HydrogenTurbineData(UnitModelBlockData):
    """
    Simple 0D hydrogen unit model that consists of three units:
    Compressor --> Stoichiometric Reaactor --> Turbine.
    """
    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""HydrogenTurbine does not support dynamic models,
        thus this must be False."""))
    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag",
        doc="""HydrogenTurbine does not support dynamic models,
        thus this must be False."""))
    CONFIG.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}"""))
    CONFIG.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))
    CONFIG.declare("reaction_package", ConfigValue(
        default=None,
        domain=is_reaction_parameter_block,
        description="Reaction package to use for control volume",
        doc="""Reaction parameter object used to define reaction calculations,
**default** - None.
**Valid values:** {
**None** - no reaction package,
**ReactionParameterBlock** - a ReactionParameterBlock object.}"""))
    CONFIG.declare("reaction_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing reaction packages",
        doc="""A ConfigBlock with arguments to be passed to a reaction block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see reaction package for documentation.}"""))

    def build(self):

        # Call UnitModel.build to setup dynamics
        super(HydrogenTurbineData, self).build()

        self.compressor = Compressor(
            default={"property_package": self.config.property_package})

        self.stoic_reactor = StoichiometricReactor(
            default={"property_package": self.config.property_package,
                     "reaction_package": self.config.reaction_package,
                     "has_heat_of_reaction": True,
                     "has_heat_transfer": False,
                     "has_pressure_change": False})

        self.turbine = Turbine(
            default={"property_package": self.config.property_package})

        # Declare var for reactor conversion
        self.stoic_reactor.conversion = Var(initialize=0.75, bounds=(0, 1))

        stoic_reactor_in = self.stoic_reactor.control_volume.properties_in[0.0]
        stoic_reactor_out = self.stoic_reactor.control_volume.properties_out[0.0]

        self.stoic_reactor.conv_constraint = Constraint(
            expr=self.stoic_reactor.conversion *
            stoic_reactor_in.flow_mol_comp["hydrogen"] ==
            (stoic_reactor_in.flow_mol_comp["hydrogen"] -
             stoic_reactor_out.flow_mol_comp["hydrogen"]))

        # Connect arcs
        self.comp_to_reactor = Arc(
            source=self.compressor.outlet,
            destination=self.stoic_reactor.inlet)
        self.reactor_to_turbine = Arc(
            source=self.stoic_reactor.outlet,
            destination=self.turbine.inlet)

        TransformationFactory("network.expand_arcs").apply_to(self)

    def initialize_build(self, state_args=None,
                   solver=None, optarg=None, outlvl=idaeslog.NOTSET):

        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        init_log.info_low("Starting initialization...")

        self.compressor.initialize(state_args=state_args, outlvl=outlvl)
        propagate_state(self.comp_to_reactor)

        self.stoic_reactor.initialize(outlvl=outlvl)
        propagate_state(self.reactor_to_turbine)

        self.turbine.initialize(outlvl=outlvl)
        init_log.info_low("Initialization complete")

    def report(self, **kwargs):
        self.compressor.report()
        self.stoic_reactor.report()
        self.turbine.report()
