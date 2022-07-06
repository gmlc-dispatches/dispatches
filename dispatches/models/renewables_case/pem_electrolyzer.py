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
import pandas as pd
import textwrap
# Import Pyomo libraries
from pyomo.environ import Var, Reals, value, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.util.calc_var_value import calculate_variable_from_constraint

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.tables import stream_table_dataframe_to_string, create_stream_table_dataframe
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              number_variables,
                                              number_activated_constraints,
                                              number_activated_blocks)
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)


def _make_pem_electrolyzer_config_block(config):
    config.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""PEM Electrolyzer does not support dynamic models, thus this must be
False."""))
    config.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag",
        doc="""Gibbs reactors do not have defined volume, thus this must be
    False."""))
    config.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PropertyParameterObject** - a PropertyParameterBlock object.}"""))
    config.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))


@declare_process_block_class("PEM_Electrolyzer", doc="Simple 0D proton-exchange membrane electrolyzer model.")
class PEMElectrolyzerData(UnitModelBlockData):
    """
    Simple 0D proton-exchange membrane electrolyzer model.
    Unit model to convert electricity and water into H2 gas.
    """
    CONFIG = ConfigBlock()
    _make_pem_electrolyzer_config_block(CONFIG)

    def build(self):
        """Building model

        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(PEMElectrolyzerData, self).build()

        self.electricity_to_mol = Var(self.flowsheet().config.time,
                                      domain=Reals,
                                      initialize=0.0,
                                      doc="Efficiency",
                                      units=pyunits.mol/pyunits.kW/pyunits.second)

        self.electricity = Var(self.flowsheet().config.time,
                               domain=Reals,
                               initialize=0.0,
                               doc="Electricity into control volume",
                               units=pyunits.kW)

        self.electricity_in = Port(noruleinit=True, doc="A port for electricity flow")
        self.electricity_in.add(self.electricity, "electricity")

        self.outlet_state = self.config.property_package.build_state_block(self.flowsheet().config.time,
                                                                           default=self.config.property_package_args)
        self.add_outlet_port(name="outlet",
                             block=self.outlet_state,
                             doc="H2 out of electrolyzer")

        @self.Constraint(self.flowsheet().config.time)
        def efficiency_curve(b, t):
            return pyunits.convert(b.outlet.flow_mol[t], to_units=pyunits.mol / pyunits.s) == b.electricity[t] * \
                   b.electricity_to_mol[t]

    def _get_performance_contents(self, time_point=0):
        return {"vars": {"Efficiency": self.electricity_to_mol[time_point]}}

    def initialize_build(self, solver=None, optarg=None, outlvl=idaeslog.NOTSET, **kwargs):
        for t in self.flowsheet().config.time:
            calculate_variable_from_constraint(self.outlet.flow_mol[t],
                                               self.efficiency_curve[t])

        self.outlet_state.initialize(hold_state=False,
                                     solver=solver,
                                     optarg=optarg,
                                     outlvl=outlvl)

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
        stream_table = create_stream_table_dataframe({"Outlet": self.outlet}, time_point=time_point)
        stream_table.loc['Electricity'] = pd.Series({'Outlet': '-'})
        stream_table.insert(0, "Inlet", ["-", "-", "-", "-", value(self.electricity[time_point])])

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
