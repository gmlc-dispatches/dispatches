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
import sys
from pandas import DataFrame
from collections import OrderedDict
import textwrap
# Import Pyomo libraries
from pyomo.environ import NonNegativeReals, Var, Expression, SolverFactory, Reference, value, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData)
from idaes.core.util import from_json, to_json, StoreSpec
from idaes.core.util.config import ListOf
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.tables import stream_table_dataframe_to_string
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              number_variables,
                                              number_activated_constraints,
                                              number_activated_blocks)

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
            domain=ListOf(str),
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
    CONFIG.declare(
        "add_split_fraction_vars",
        ConfigValue(
            domain=bool,
            default=False,
            description="Add split fraction variables. Set it to True if these variables are needed"
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

        @self.Constraint(time, doc="Split constraint")
        def sum_split(b, t):
            return b.electricity[t] == sum(getattr(b, o + "_elec")[t] for o in b.outlet_list)

        if self.config.add_split_fraction_vars:
            self.split_fraction = Var(self.outlet_list,
                                        time,
                                        bounds=(0, 1),
                                        initialize=1.0 / len(self.outlet_list),
                                        doc="Split fractions for outlet streams")

            @self.Constraint(time, self.outlet_list, doc="Split fraction definition")
            def split_fraction_definition(b, t, o):
                outlet_obj = getattr(b, o + "_elec")
                return outlet_obj[t] == b.split_fraction[o, t] * b.electricity[t]
        else:
            self.split_fraction = Expression(self.outlet_list,
                                             time,
                                             rule=lambda b, o, t: getattr(b, o + "_elec")[t] / b.electricity[t],
                                             doc="Split fractions for outlet streams")


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

    def initialize_build(self, **kwargs):
        # store original state
        sp = StoreSpec.value_isfixed_isactive(only_fixed=True)
        istate = to_json(self, return_dict=True, wts=sp)

        if self.config.add_split_fraction_vars:
            outlet_vars = [Reference(self.split_fraction[o, :]) for o in self.outlet_list]
        else:
            outlet_vars = [getattr(self, o + "_elec") for o in self.outlet_list]

        # fix or unfix electricity flows so n - 1 are fixed
        for t in self.flowsheet().config.time:
            # see how many electricity flows are fixed
            n = sum(1 for v in outlet_vars if v[t].fixed)
            # if number of outlets - 1 we're good
            if n == len(self.outlet_list) - 1:
                continue
            # if too many are fixed, unfix the first, generally assume that is the main flow
            elif n == len(self.outlet_list):
                outlet_vars[0][t].unfix()
            # if not enough fixed, start fixing from the back until there are are enough
            else:
                for v in reversed(outlet_vars):
                    if not v[t].fixed:
                        v[t].fix()
                        n += 1
                    if n == len(self.outlet_list) - 1:
                        break
        self.electricity.fix()
        
        assert degrees_of_freedom(self) == 0

        solver = "ipopt"
        if "solver" in kwargs:
            solver = kwargs["solver"]

        opt = SolverFactory(solver)
        opt.solve(self)

        from_json(self, sd=istate, wts=sp)

    def report(self, time_point=0, dof=False, ostream=None, prefix=""):
        time_point = float(time_point)

        if ostream is None:
            ostream = sys.stdout

        # Get DoF and model stats
        if dof:
            dof_stat = degrees_of_freedom(self)
            nv = number_variables(self)
            nc = number_activated_constraints(self)
            nb = number_activated_blocks(self)

        # Get stream table
        stream_attributes = OrderedDict()
        stream_attributes["Inlet"] = {'electricity': value(self.electricity[time_point])}
        stream_attributes["Outlet"] = {}
        for outlet in self.outlet_list:
            stream_attributes["Outlet"][outlet] = value(getattr(self, outlet + "_elec")[time_point])

        stream_table = DataFrame.from_dict(stream_attributes, orient="columns")

        # Set model type output
        if hasattr(self, "is_flowsheet") and self.is_flowsheet:
            model_type = "Flowsheet"
        else:
            model_type = "Unit"

        # Write output
        max_str_length = 84
        tab = " " * 4
        ostream.write("\n" + "=" * max_str_length + "\n")

        lead_str = f"{prefix}{model_type} : {self.name}"
        trail_str = f"Time: {time_point}"
        mid_str = " " * (max_str_length - len(lead_str) - len(trail_str))
        ostream.write(lead_str + mid_str + trail_str)

        if dof:
            ostream.write("\n" + "=" * max_str_length + "\n")
            ostream.write(f"{prefix}{tab}Local Degrees of Freedom: {dof_stat}")
            ostream.write('\n')
            ostream.write(f"{prefix}{tab}Total Variables: {nv}{tab}"
                          f"Activated Constraints: {nc}{tab}"
                          f"Activated Blocks: {nb}")

        if stream_table is not None:
            ostream.write("\n" + "-" * max_str_length + "\n")
            ostream.write(f"{prefix}{tab}Stream Table")
            ostream.write('\n')
            ostream.write(
                textwrap.indent(
                    stream_table_dataframe_to_string(stream_table),
                    prefix + tab))
        ostream.write("\n" + "=" * max_str_length + "\n")