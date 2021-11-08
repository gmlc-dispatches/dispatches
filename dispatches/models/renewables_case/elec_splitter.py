#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
# Import Pyomo libraries
from pyomo.environ import NonNegativeReals, Var, Reals, SolverFactory, value, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (Component,
                        ControlVolume0DBlock,
                        declare_process_block_class,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.util import from_json, to_json, StoreSpec
from idaes.core.util.config import list_of_strings
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import degrees_of_freedom

import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ElectricalSplitter", doc="Splits electricity flow into outlet stream")
class ElectricalSplitterData(UnitModelBlockData):
    """
    Unit model to split a electricity from a single inlet into multiple outlets based on split fractions
    """
    CONFIG = ConfigBlock()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False"
        )
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False"
        )
    )
    CONFIG.declare(
        "outlet_list",
        ConfigValue(
            domain=list_of_strings,
            description="List of outlet names",
            doc="""A list containing names of outlets,
                **default** - None.
                **Valid values:** {
                **None** - use num_outlets argument,
                **list** - a list of names to use for outlets.}"""
        )
    )
    CONFIG.declare(
        "num_outlets",
        ConfigValue(
            domain=int,
            description="Number of outlets to unit",
            doc="""Argument indicating number (int) of outlets to construct,
                not used if outlet_list arg is provided,
                **default** - None.
                **Valid values:** {
                **None** - use outlet_list arg instead, or default to 2 if neither argument
                provided,
                **int** - number of outlets to create (will be named with sequential integers
                from 1 to num_outlets).}"""
        )
    )

    def build(self):
        """

        """
        super().build()
        time = self.flowsheet().config.time

        self.create_outlets()

        self.electricity = Var(time,
                               domain=NonNegativeReals,
                               initialize=0.0,
                               doc="Electricity into control volume",
                               units=pyunits.kW)
        self.electricity_in = Port(noruleinit=True, doc="A port for electricity flow")
        self.electricity_in.add(self.electricity, "electricity")

        self.split_fraction = Var(self.outlet_list,
                                  time,
                                  bounds=(0, 1),
                                  initialize=1.0/len(self.outlet_list),
                                  doc="Split fractions for outlet streams"
        )

        @self.Constraint(time, doc="Split constraint")
        def sum_split(b, t):
            return 1 == sum(b.split_fraction[o, t] for o in b.outlet_list)

        @self.Constraint(time, self.outlet_list, doc="Electricity constraint")
        def electricity_eqn(b, t, o):
            outlet_obj = getattr(b, o + "_elec")
            return outlet_obj[t] == b.split_fraction[o, t] * b.electricity[t]

    def create_outlets(self):
        """
        Create list of outlet stream names based on config arguments.

        Returns:
            list of strings
        """
        config = self.config
        if config.outlet_list is not None and config.num_outlets is not None:
            # If both arguments provided and not consistent, raise Exception
            if len(config.outlet_list) != config.num_outlets:
                raise ConfigurationError(
                    "{} ElectricalSplitter provided with both outlet_list and "
                    "num_outlets arguments, which were not consistent ("
                    "length of outlet_list was not equal to num_outlets). "
                    "Please check your arguments for consistency, and "
                    "note that it is only necessry to provide one of "
                    "these arguments.".format(self.name)
                )
        elif (config.outlet_list is None and config.num_outlets is None):
            # If no arguments provided for outlets, default to num_outlets = 2
            config.num_outlets = 2

        # Create a list of names for outlet StateBlocks
        if config.outlet_list is not None:
            outlet_list = self.config.outlet_list
        else:
            outlet_list = [
                "outlet_{}".format(n) for n in range(1, config.num_outlets + 1)
            ]
        self.outlet_list = outlet_list

        for p in self.outlet_list:
            outlet_obj = Var(self.flowsheet().config.time,
                                             domain=NonNegativeReals,
                                             initialize=0.0,
                                             doc="Electricity at outlet {}".format(p),
                                             units=pyunits.kW)
            setattr(self, p + "_elec", outlet_obj)
            outlet_port = Port(noruleinit=True, doc="Outlet {}".format(p))
            outlet_port.add(getattr(self, p + "_elec"), "electricity")
            setattr(self, p + "_port", outlet_port)

    def initialize(self, **kwargs):
        # store original state
        sp = StoreSpec.value_isfixed_isactive(only_fixed=True)
        istate = to_json(self, return_dict=True, wts=sp)

        # check for fixed outlet flows and use them to calculate fixed split
        # fractions
        for t in self.flowsheet().config.time:
            for o in self.outlet_list:
                elec_obj = getattr(self, o + "_elec")
                if elec_obj[t].fixed:
                    self.split_fraction[o, t].fix(
                        value(elec_obj[t] / self.electricity[t]))

        # fix or unfix split fractions so n - 1 are fixed
        for t in self.flowsheet().config.time:
            # see how many split fractions are fixed
            n = sum(1 for o in self.outlet_list if self.split_fraction[o, t].fixed)
            # if number of outlets - 1 we're good
            if n == len(self.outlet_list) - 1:
                continue
            # if too many are fixed, unfix the first, generally assume that is
            # the main flow, and is the calculated split fraction
            elif n == len(self.outlet_list):
                self.split_fraction[self.outlet_list[0], t].unfix()
            # if not enough fixed, start fixing from the back until there are
            # are enough
            else:
                for o in reversed(self.outlet_list):
                    if not self.split_fraction[o, t].fixed:
                        self.split_fraction[o, t].fix()
                        n += 1
                    if n == len(self.outlet_list) - 1:
                        break

        self.electricity.fix()
        for o in self.outlet_list:
            getattr(self, o + "_port").unfix()
        assert degrees_of_freedom(self) == 0

        solver = "ipopt"
        if "solver" in kwargs:
            solver = kwargs["solver"]

        opt = SolverFactory(solver)
        opt.solve(self)

        from_json(self, sd=istate, wts=sp)
