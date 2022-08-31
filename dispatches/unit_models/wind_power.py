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
from pyomo.environ import Var, Param, NonNegativeReals, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In, ListOf

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        UnitModelBlockData)
from idaes.core.util.initialization import solve_indexed_blocks
import idaes.logger as idaeslog

import PySAM.Windpower as wind

_log = idaeslog.getLogger(__name__)


# TODO: move to idaes/core/util/config.py
def list_of_list_of_floats(arg):
    '''Domain validator for lists of floats

    Args:
        arg : argument to be cast to list of floats and validated

    Returns:
        List of list of floats
    '''
    try:
        lst = [[float(i) for i in j] for j in arg]
    except TypeError:
        lst = [ListOf(float)(arg), ]
    return lst


def dict_of_list_of_list_of_floats(arg):
    dic = dict()
    for k, v in arg.items():
        dic[k] = list_of_list_of_floats(v)
    return dic


@declare_process_block_class("Wind_Power", doc="Wind plant using turbine powercurve and resource data")
class WindpowerData(UnitModelBlockData):
    """
    Wind plant using turbine powercurve and resource data.
    Unit model to convert wind resource into electricity.
    """
    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""Wind plant does not support dynamic models, thus this must be False."""))
    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag",
        doc="""Wind plant does not have defined volume, thus this must be False."""))
    CONFIG.declare("resource_speed", ConfigValue(
        default=None,
        domain=ListOf(float),
        description="List: Wind speed meters per sec",
        doc="For each time in flowsheet's time set, wind speed [m/s]"))

    CONFIG.declare("resource_probability_density", ConfigValue(
        default=None,
        domain=dict_of_list_of_list_of_floats,
        description="Dictionary of Time: List of (wind speed meters per sec, wind degrees clockwise from north, probability)",
        doc="For each time in flowsheet's time set, a probability density function of "
            "Wind speed [m/s] and Wind direction [degrees clockwise from N]"))
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
                                     doc="Rated system capacity of wind farm",
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

    @staticmethod
    def setup_atb_turbine():
        """
        """
        wind_simulation = wind.default("WindpowerSingleowner")

        # Use ATB Turbine 2018 Market Average
        wind_simulation.Turbine.wind_turbine_hub_ht = 110
        wind_simulation.Turbine.wind_turbine_rotor_diameter = 116
        wind_simulation.Turbine.wind_turbine_powercurve_powerout = [0, 0, 0, 40.500000, 177.700000, 403.900000, 737.600000, 1187.200000, 1771.100000, 2518.600000, 3448.400000, 4562.500000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 0, 0]
        wind_simulation.Turbine.wind_turbine_powercurve_windspeeds = [
            1 * i for i in range(len(wind_simulation.Turbine.wind_turbine_powercurve_powerout))]

        # Use a single turbine, do not model wake effects
        wind_simulation.Farm.wind_farm_xCoordinates = [0]
        wind_simulation.Farm.wind_farm_yCoordinates = [0]
        wind_simulation.Farm.system_capacity = max(wind_simulation.Turbine.wind_turbine_powercurve_powerout)
        return wind_simulation

    def setup_resource(self):
        """
        """
        if self.config.resource_probability_density is not None:
            for time in list(self.flowsheet().config.time.data()):
                if time not in self.config.resource_probability_density.keys():
                    raise ValueError("'resource_probability_density' must contain data for time {}".format(time))

            for time, resource in self.config.resource_probability_density.items():
                if abs(sum(r[2] for r in resource) - 1) > 1e-3:
                    raise ValueError("Error in 'resource_probability_density' for time {}: Probabilities of "
                                     "Wind Speed and Direction Probability Density Function must sum to 1")
                if len(resource) != 1:
                    raise NotImplementedError
                wind_simulation = self.setup_atb_turbine()
                wind_simulation.Resource.wind_resource_model_choice = 2
                wind_simulation.Resource.wind_resource_distribution = resource
                wind_simulation.execute(0)
                self.capacity_factor[time].set_value(wind_simulation.Outputs.capacity_factor / 100.)
        elif self.config.resource_speed is not None:
            if len(self.config.resource_speed) < len(self.flowsheet().config.time.data()):
                raise ValueError(f"'resource_speed' list length must be at least {len(self.flowsheet().config.time.data())}")
            for time, speed in zip(list(self.flowsheet().config.time.data()), self.config.resource_speed):
                wind_simulation = self.setup_atb_turbine()
                wind_simulation.Resource.wind_resource_model_choice = 1
                wind_simulation.Resource.weibull_k_factor = 100
                wind_simulation.Resource.weibull_reference_height = 110
                wind_simulation.Resource.weibull_wind_speed = speed
                wind_simulation.execute(0)
                self.capacity_factor[time].set_value(wind_simulation.Outputs.capacity_factor / 100.)
        elif self.config.capacity_factor:
            if not len(self.config.capacity_factor) >= len(self.flowsheet().config.time.data()):
                raise ValueError("Config with 'capacity_factor' must provide data per time step")
                
            for n, time in enumerate(self.flowsheet().config.time.data()):
                self.capacity_factor[time].set_value(self.config.capacity_factor[n])   
        else:
            raise ValueError("Config with 'resource_probability_density' or `capacity_factor` must be provided using `default` argument")

    def initialize_build(self, **kwargs):
        for t in self.flowsheet().config.time:
            self.electricity[t].set_value(self.system_capacity * self.capacity_factor[t])
