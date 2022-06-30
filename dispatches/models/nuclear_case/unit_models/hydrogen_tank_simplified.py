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


# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.environ import Var, NonNegativeReals

# Import IDAES cores
from idaes.core import declare_process_block_class, UnitModelBlockData
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog

__author__ = "Radhakrishna Tumbalam Gooty"


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("SimpleHydrogenTank")
class SimpleHydrogenTankData(UnitModelBlockData):
    """
    Simplified tank model where energy balance is ignored
    """

    CONFIG = ConfigBlock()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Translator blocks are always steady-state.""",
        ),
    )
    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag - must be False",
        doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. Hydrogen tank model uses custom equations for holdup.""")
    )
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for incoming stream",
            doc="""Property parameter object used to define property
calculations for the incoming stream,
**default** - None.
**Valid values:** {
**PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property package "
            "of the incoming stream",
            doc="""A ConfigBlock with arguments to be passed to the property
block associated with the incoming stream,
**default** - None.
**Valid values:** {
see property package for documentation.}""",
        ),
    )

    def build(self):
        """
        Begin building model.
        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(SimpleHydrogenTankData, self).build()

        # Add State Blocks
        self.properties_in = self.config.property_package.build_state_block(
            self.flowsheet().time,
            doc="Material properties in incoming stream",
            default={
                "defined_state": True,
                "has_phase_equilibrium": False,
                **self.config.property_package_args,
            },
        )

        self.properties_out_pipeline = self.config.property_package.build_state_block(
            self.flowsheet().time,
            doc="Material properties in outlet stream to pipeline",
            default={
                "defined_state": True,
                "has_phase_equilibrium": False,
                **self.config.property_package_args,
            },
        )

        self.properties_out_turbine = self.config.property_package.build_state_block(
            self.flowsheet().time,
            doc="Material properties in outlet stream to turbine",
            default={
                "defined_state": True,
                "has_phase_equilibrium": False,
                **self.config.property_package_args,
            },
        )

        # Add outlet port
        self.add_port(name="inlet",
                      block=self.properties_in,
                      doc="Inlet Port")
        self.add_port(name="outlet_to_turbine",
                      block=self.properties_out_turbine,
                      doc="Outlet Port to Hydrogen Turbine")
        self.add_port(name="outlet_to_pipeline",
                      block=self.properties_out_pipeline,
                      doc="Outlet Port to Pipeline")

        # Keep the temperature and pressure the same for inlets and outlets
        @self.Constraint(self.flowsheet().time)
        def eq_temperature_1(blk, t):
            return (
                blk.properties_in[t].temperature ==
                blk.properties_out_turbine[t].temperature
            )

        @self.Constraint(self.flowsheet().time)
        def eq_temperature_2(blk, t):
            return (
                blk.properties_in[t].temperature ==
                blk.properties_out_pipeline[t].temperature
            )

        @self.Constraint(self.flowsheet().time)
        def eq_pressure_1(blk, t):
            return (
                blk.properties_in[t].pressure ==
                blk.properties_out_turbine[t].pressure
            )

        @self.Constraint(self.flowsheet().time)
        def eq_pressure_2(blk, t):
            return (
                blk.properties_in[t].pressure ==
                blk.properties_out_pipeline[t].pressure
            )

        # Get units from property package
        units = self.config.property_package.\
            get_metadata().get_derived_units

        # Define holdup variable
        self.tank_holdup_previous = Var(self.flowsheet().time,
                                        within=NonNegativeReals,
                                        units=units("amount"))
        self.tank_holdup = Var(self.flowsheet().time,
                               within=NonNegativeReals,
                               units=units("amount"))
        self.dt = Var(self.flowsheet().time,
                      initialize=3600,
                      within=NonNegativeReals,
                      units=units("time"))

        # Material balance
        @self.Constraint(self.flowsheet().time)
        def tank_material_balance(blk, t):
            return (
                blk.tank_holdup[t] - blk.tank_holdup_previous[t] ==
                blk.dt[t] * (+ blk.properties_in[t].flow_mol
                             - blk.properties_out_pipeline[t].flow_mol
                             - blk.properties_out_turbine[t].flow_mol)
            )

    def initialize_build(
        blk,
        state_args_in=None,
        state_args_out=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        This method calls the initialization method of the state blocks.
        Keyword Arguments:
            state_args_in : a dict of arguments to be passed to the inlet
                            property package (to provide an initial state for
                            initialization (see documentation of the specific
                            property package) (default = None).
            state_args_out : a dict of arguments to be passed to the outlet
                             property package (to provide an initial state for
                             initialization (see documentation of the specific
                             property package) (default = None).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)
        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize state block
        flags = blk.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_in,
            hold_state=True,
        )

        blk.properties_out_turbine.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_out,
        )

        blk.properties_out_pipeline.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_out,
        )

        if degrees_of_freedom(blk) != 0:
            raise AssertionError(f"Initialization of {blk.name} is unsuccessful. "
                                 f"Degrees of freedom was not zero. Please provide "
                                 f"sufficient number of constraints linking the state "
                                 f"variables between the two state blocks.")

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)

        init_log.info("Initialization Complete {}."
                      .format(idaeslog.condition(res)))

        blk.properties_in.release_state(flags=flags, outlvl=outlvl)
