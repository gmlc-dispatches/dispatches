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

_log = idaeslog.getLogger(__name__)


@declare_process_block_class("BatteryStorage", doc="Wind plant using turbine powercurve and resource data")
class BatteryStorageData(UnitModelBlockData):
    """
    Wind plant using turbine powercurve and resource data.
    Unit model to convert wind resource into electricity.
    """
    CONFIG = ConfigBlock()
    CONFIG.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""Battery does not support dynamic models, thus this must be False."""))
    CONFIG.declare("has_holdup", ConfigValue(
        default=False,
        domain=In([False]),
        description="Holdup construction flag",
        doc="""Battery does not have defined volume, thus this must be False."""))

    def build(self):
        """Building model

        Args:
            None
        Returns:
            None
        """
        super(BatteryStorageData, self).build()

        self.dt = Param(within=NonNegativeReals,
                        initialize=1,
                        doc="Time step",
                        units=pyunits.hr)

        self.nameplate_power = Var(within=NonNegativeReals,
                                   initialize=0.0,
                                   doc="Nameplate power of battery energy storage",
                                   units=pyunits.kW)

        self.nameplate_energy = Var(within=NonNegativeReals,
                                    initialize=0.0,
                                    doc="Nameplate energy of battery energy storage",
                                    units=pyunits.kWh)

        self.charging_eta = Param(within=NonNegativeReals,
                                  mutable=True,
                                  initialize=1,
                                  doc="Charging efficiency, (0, 1]")

        self.discharging_eta = Param(within=NonNegativeReals,
                                     mutable=True,
                                     initialize=1,
                                     doc="Discharging efficiency, (0, 1]")

        self.degradation_rate = Param(within=NonNegativeReals,
                                      mutable=True,
                                      initialize=0.8/3800,
                                      doc="Degradation rate, [0, 2.5e-3]",
                                      units=pyunits.hr/pyunits.hr)

        self.initial_state_of_charge = Var(within=NonNegativeReals,
                                           initialize=0.0,
                                           doc="State of charge at t - 1, [0, self.nameplate_energy]",
                                           units=pyunits.kWh)

        self.initial_energy_throughput = Var(within=NonNegativeReals,
                                             initialize=0.0,
                                             doc="Cumulative energy throughput at t - 1",
                                             units=pyunits.kWh)

        self.state_of_charge = Var(self.flowsheet().config.time,
                                   within=NonNegativeReals,
                                   initialize=0.0,
                                   doc="State of charge (energy), [0, self.nameplate_energy]",
                                   units=pyunits.kWh)

        self.elec_in = Var(self.flowsheet().config.time,
                             within=NonNegativeReals,
                             initialize=0.0,
                             doc="Energy in",
                             units=pyunits.kW)

        self.elec_out = Var(self.flowsheet().config.time,
                              within=NonNegativeReals,
                              initialize=0.0,
                              doc="Energy out",
                              units=pyunits.kW)

        self.energy_throughput = Var(self.flowsheet().config.time,
                                     within=NonNegativeReals,
                                     initialize=0.0,
                                     doc="Cumulative energy throughput",
                                     units=pyunits.kWh)

        self.power_in = Port(noruleinit=True, doc="A port for electricity inflow")
        self.power_in.add(self.elec_in, "electricity")

        self.power_out = Port(noruleinit=True, doc="A port for electricity outflow")
        self.power_out.add(self.elec_out, "electricity")

        @self.Constraint(self.flowsheet().config.time)
        def state_evolution(b, t):
            if t == 0:
                return b.state_of_charge[t] == b.initial_state_of_charge + (
                        b.charging_eta * b.dt * b.elec_in[t]
                        - b.dt / b.discharging_eta * b.elec_out[t])
            return b.state_of_charge[t] == b.state_of_charge[t-1] + (
                    b.charging_eta * b.dt * b.elec_in[t]
                    - b.dt / b.discharging_eta * b.elec_out[t])

        @self.Constraint(self.flowsheet().config.time)
        def accumulate_energy_throughput(b, t):
            if t == 0:
                return b.energy_throughput[t] == b.initial_energy_throughput + b.dt * (b.elec_in[t] + b.elec_out[t]) / 2
            return b.energy_throughput[t] == b.energy_throughput[t - 1] + b.dt * (b.elec_in[t] + b.elec_out[t]) / 2

        @self.Constraint(self.flowsheet().config.time)
        def state_of_charge_bounds(b, t):
            return b.state_of_charge[t] <= b.nameplate_energy - b.degradation_rate * b.energy_throughput[t]

        @self.Constraint(self.flowsheet().config.time)
        def power_bound_in(b, t):
            return b.elec_in[t] <= b.nameplate_power

        @self.Constraint(self.flowsheet().config.time)
        def power_bound_out(b, t):
            return b.elec_out[t] <= b.nameplate_power
