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
# Import Pyomo libraries
from pyomo.environ import Var, Param, NonNegativeReals, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In, ListOf

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData)
from idaes.core.util.initialization import solve_indexed_blocks
import idaes.logger as idaeslog

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("SolarPV", doc="Soar PV plant using capacity factors")
class SolarPVData(UnitModelBlockData):
    """
    Solar PV plant using resource data.
    Unit model to convert Solar PV resource into electricity.
    """
    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""Solar PV plant does not support dynamic models, thus this must be False."""))
    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag",
        doc="""Solar PV plant does not have defined volume, thus this must be False."""))
    CONFIG.declare("capacity_factor", ConfigValue(
        default=None,
        domain=ListOf(float),
        description="Capacity Factors per Timestep",
        doc="Capacity Factor in a list for each Timestep in model"))

    def build(self):
        """Building model

        Args:
            None
        Returns:
            None
        """
        super().build()

        self.system_capacity = Var(within=NonNegativeReals,
                                     initialize=0.0,
                                     doc="Rated system capacity of Solar PV farm",
                                     units=pyunits.kW)

        self.capacity_factor = Param(self.flowsheet().config.time,
                                     within=NonNegativeReals,
                                     mutable=True,
                                     initialize=0.0,
                                     doc="Ratio of power output to rated capacity, on annual or time series basis",
                                     units=pyunits.kW/pyunits.kW)

        self.electricity = Var(self.flowsheet().config.time,
                               within=NonNegativeReals,
                               initialize=0.0,
                               doc="Electricity production",
                               units=pyunits.kW)

        self.electricity_out = Port(noruleinit=True, doc="A port for electricity flow")
        self.electricity_out.add(self.electricity, "electricity")

        @self.Constraint(self.flowsheet().config.time)
        def elec_from_capacity_factor(b, t):
            return b.electricity[t] <= self.system_capacity * self.capacity_factor[t]

        self.setup_resource()

    def _get_performance_contents(self, time_point=0):
        return {"vars": {"Electricity": self.electricity[time_point].value}}

    def setup_resource(self):
        """
        """
        if not self.config.capacity_factor:
            raise ValueError("Config with `capacity_factor` must be provided using `default` argument")

        if not len(self.config.capacity_factor) >= len(self.flowsheet().config.time.data()):
            raise ValueError("Config with 'capacity_factor' must provide data per time step")
            
        for n, time in enumerate(self.flowsheet().config.time.data()):
            self.capacity_factor[time].set_value(self.config.capacity_factor[n])   

    def initialize_build(self, **kwargs):
        for t in self.flowsheet().config.time:
            self.electricity[t].set_value(self.system_capacity * self.capacity_factor[t])
