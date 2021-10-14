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


@declare_process_block_class("BatteryStorage", doc="Simple battery model")
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
        This model does not use the flowsheet's time domain. Instead, it only models a single timestep, with initial
        conditions provided by `initial_state_of_charge` and `initial_energy_throughput`. The model calculates change
        in stored energy across a single time step using the power flow variables, `power_in` and `power_out`, and
        the `dr_hr` parameter.
        Args:
            None
        Returns:
            None
        """
        super().build()

        # Design variables and parameters
        self.nameplate_power = Var(within=NonNegativeReals,
                                   initialize=0.0,
                                   bounds=(0, 1e6),
                                   doc="Nameplate power of battery energy storage",
                                   units=pyunits.kW)

        self.nameplate_energy = Var(within=NonNegativeReals,
                                    initialize=0.0,
                                    bounds=(0, 1e7),
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

        # Initial conditions
        self.initial_state_of_charge = Var(within=NonNegativeReals,
                                           initialize=0.0,
                                           doc="State of charge at t - 1, [0, self.nameplate_energy]",
                                           units=pyunits.kWh)

        self.initial_energy_throughput = Var(within=NonNegativeReals,
                                             initialize=0.0,
                                             doc="Cumulative energy throughput at t - 1",
                                             units=pyunits.kWh)

        # Power flows and energy storage
        self.dt = Param(within=NonNegativeReals,
                        initialize=1,
                        doc="Time step for converting between electricity power flows and stored energy",
                        units=pyunits.hr)

        self.elec_in = Var(within=NonNegativeReals,
                           initialize=0.0,
                           doc="Energy in",
                           units=pyunits.kW)

        self.elec_out = Var(within=NonNegativeReals,
                            initialize=0.0,
                            doc="Energy out",
                            units=pyunits.kW)

        self.state_of_charge = Var(within=NonNegativeReals,
                                   initialize=0.0,
                                   doc="State of charge (energy), [0, self.nameplate_energy]",
                                   units=pyunits.kWh)

        self.energy_throughput = Var(within=NonNegativeReals,
                                     initialize=0.0,
                                     doc="Cumulative energy throughput",
                                     units=pyunits.kWh)

        # Ports
        self.power_in = Port(noruleinit=True, doc="A port for electricity inflow")
        self.power_in.add(self.elec_in, "electricity")

        self.power_out = Port(noruleinit=True, doc="A port for electricity outflow")
        self.power_out.add(self.elec_out, "electricity")

        @self.Constraint()
        def state_evolution(b):
            return b.state_of_charge == b.initial_state_of_charge + (
                    b.charging_eta * b.dt * b.elec_in
                    - b.dt / b.discharging_eta * b.elec_out)

        @self.Constraint()
        def accumulate_energy_throughput(b):
            return b.energy_throughput == b.initial_energy_throughput + b.dt * (b.elec_in + b.elec_out) / 2

        @self.Constraint()
        def state_of_charge_bounds(b):
            return b.state_of_charge <= b.nameplate_energy - b.degradation_rate * b.energy_throughput

        @self.Constraint()
        def power_bound_in(b):
            return b.elec_in <= b.nameplate_power

        @self.Constraint()
        def power_bound_out(b):
            return b.elec_out <= b.nameplate_power
