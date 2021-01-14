# Import Pyomo libraries
from pyomo.environ import Var, Param, NonNegativeReals, units as pyunits
from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (Component,
                        ControlVolume0DBlock,
                        declare_process_block_class,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        MaterialBalanceType,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.util.config import list_of_floats
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
        lst = [list_of_floats(arg), ]
    return lst


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
    CONFIG.declare("resource_probability_density", ConfigValue(
        default=None,
        domain=list_of_list_of_floats,
        description="List of (wind meters per sec, wind degrees clockwise from north, probability)",
        doc="Probability Density Function of Wind speed [m/s] and Wind direction [degrees clockwise from N]"))
    CONFIG.declare("resource_timeseries", ConfigValue(
        default=None,
        domain=list_of_list_of_floats,
        description=""
    ))

    def build(self):
        """Building model

        Args:
            None
        Returns:
            None
        """
        super(WindpowerData, self).build()

        self.system_capacity = Param(within=NonNegativeReals,
                                     initialize=0.0,
                                     doc="Rated system capacity of wind farm",
                                     units=pyunits.kW/pyunits.kW)

        self.capacity_factor = Param(within=NonNegativeReals,
                                     initialize=0.0,
                                     doc="Ratio of power output to rated capacity",
                                     units=pyunits.kW/pyunits.kW)

        self.electricity = Var(self.flowsheet().config.time,
                               within=NonNegativeReals,
                               initialize=0.0,
                               doc="Electricity into control volume",
                               units=pyunits.kW)

        self.outlet = Port(noruleinit=True, doc="A port for electricity flow")
        self.outlet.add(self.electricity, "electricity")

        @self.Constraint(self.flowsheet().config.time)
        def efficiency_curve(b, t):
            return b.electricity[t] == self.system_capacity * self.capacity_factor

    def _get_performance_contents(self, time_point=0):
        return {"vars": {"Electricity": self.electricity[time_point]}}

    def initialize(self, **kwargs):
        self.setup_turbine_power()

    def setup_turbine_power(self):
        wind_simulation = wind.default("WindpowerSingleowner")

        # Use ATB Turbine 2018 Market Average
        wind_simulation.Turbine.wind_turbine_hub_ht = 88
        wind_simulation.Turbine.wind_turbine_rotor_diameter = 116
        wind_simulation.Turbine.wind_turbine_powercurve_windspeeds = [0.25 * i for i in range(161)]
        wind_simulation.Turbine.wind_turbine_powercurve_powerout = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 78, 104, 133, 167, 204, 246, 293, 345, 402, 464, 532, 606, 686,
            772, 865, 965, 1072, 1186, 1308, 1438, 1576, 1723, 1878, 2042, 2215, 2397, 2430, 2430, 2430, 2430, 2430,
            2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430,
            2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430,
            2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430, 2430,
            2430, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Use a single turbine, do not model wake effects
        wind_simulation.Farm.wind_farm_xCoordinates = [0]
        wind_simulation.Farm.wind_farm_yCoordinates = [0]
        wind_simulation.Farm.system_capacity = max(wind_simulation.Turbine.wind_turbine_powercurve_powerout)

        if self.config.resource_timeseries:
            raise NotImplementedError
        elif self.config.resource_probability_density:
            if abs(sum(r[2] for r in self.config.resource_probability_density) - 1) > 1e-3:
                raise ValueError("Probabilities of Wind Speed and Direction Probability Density Function must be 1")
            wind_simulation.Resource.wind_resource_model_choice = 2
            wind_simulation.Resource.wind_resource_distribution = self.config.resource_probability_density
            wind_simulation.execute(0)
            self.capacity_factor = wind_simulation.Outputs.annual_energy / 8760 / wind_simulation.Farm.system_capacity