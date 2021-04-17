##############################################################################
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
#
##############################################################################

"""
Basic IDAES unit model for compressed hydrogen gas (chg) tank.
Uses custom material and energy balance equations.
Suitable for use with steady state flowsheets indexed with time.

An initial state block (initial_state) is defined
to pass the intial condition of tank, i.e., Pressure and Temperature.
For dynamic operations, multiple instances of this model indexed with time
can be used. For this, equality constraints must be written at
flowsheet level, i.e., 
the final state of tank (Pres & Temp) at time (t-1) ==
    intial state of tank (Pres & Temp) at time (t)

Time window of operation

This property package assumes Ideal gas behaviour of hydrogen gas.
Adiabatic operations are assumed.
TO DO: add constraints to enable isothermal operations
"""

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (Var,
                           Reals,
                           NonNegativeReals,
                           Constraint,
                           units as pyunits)
# from pyomo.network import Port
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (ControlVolume0DBlock,
                        declare_process_block_class,
                        MaterialFlowBasis,
                        MomentumBalanceType,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.constants import Constants as const
from idaes.generic_models.properties.core.eos.ideal import Ideal
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale

__author__ = "Naresh Susarla"

# Set up logger
_log = idaeslog.getLogger(__name__)

def _make_chg_tank_config_block(config):
    config.declare("dynamic", ConfigValue(
        domain=In([False]),
        default=False,
        description="Dynamic model flag - must be False",
        doc="""Compressed hydrogen gas tank model has not been tested 
for dynamic operations, thus this is set to False"""))
    config.declare("has_holdup", ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Holdup Var from the control_volume is not called.
Instead, a Var with same name is declared in this unit model to account
for holdup in the custom material and energy balances""",
        ),
    )
    config.declare("momentum_balance_type", ConfigValue(
        default=MomentumBalanceType.pressureTotal,
        domain=In(MomentumBalanceType),
        description="Momentum balance construction flag",
        doc="""Indicates what type of momentum balance should be constructed,
**default** - MomentumBalanceType.pressureTotal.
**Valid values:** {
**MomentumBalanceType.none** - exclude momentum balances,
**MomentumBalanceType.pressureTotal** - single pressure balance for material,
**MomentumBalanceType.pressurePhase** - pressure balances for each phase,
**MomentumBalanceType.momentumTotal** - single momentum balance for material,
**MomentumBalanceType.momentumPhase** - momentum balances for each phase.}"""))
    config.declare("has_pressure_change", ConfigValue(
        default=True,
        domain=In([True]),
        description="Pressure change term construction flag",
        doc="""Indicates whether terms for pressure change should be
constructed,
**default** - True. Must always be true for tank operations
**Valid values:** {
**True** - include pressure change terms}"""))
    config.declare("property_package", ConfigValue(
        default=useDefault,
        domain=is_physical_parameter_block,
        description="Property package to use for control volume",
        doc="""Property parameter object used to define property calculations,
**default** - useDefault.
**Valid values:** {
**useDefault** - use default package from parent model or flowsheet,
**PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))
    config.declare("property_package_args", ConfigBlock(
        implicit=True,
        description="Arguments to use for constructing property packages",
        doc="""A ConfigBlock with arguments to be passed to a property block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see property package for documentation.}"""))


@declare_process_block_class("CHGTank",
                             doc="Simple compressed hydrogen gas tank model")
class CHGTankData(UnitModelBlockData):
    """
    Simple compressed hydrogen gas tank model.
    Unit model to store or supply compressed hydrogen gas.
    
    """
    CONFIG = ConfigBlock()
    _make_chg_tank_config_block(CONFIG)

    def build(self):
        """Building model
        Args:
            None
        Returns:
            None
        """
        super(CHGTankData, self).build()

        # Build Control Volume
        self.control_volume = ControlVolume0DBlock(default={
            "dynamic": self.config.dynamic,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args})

        # add inlet and outlet states
        self.control_volume.add_state_blocks(has_phase_equilibrium=False)

        # add tank volume
        self.control_volume.add_geometry()

        # add pressure balance
        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=True)

        # add phase fractions
        self.control_volume._add_phase_fractions()

        # add a state block 'initial_state' for the storage tank
        # this state block is needed to compute initial material and energy
        # holdup using the property package for a given P_init and T_init
        # NOTE: there is no flow in the initial state so,
        # flow_mol state variable is fixed to 0
        self.initial_state = (
            self.config.property_package.build_state_block(
                self.flowsheet().config.time,
                doc="tank initial state",
                default=self.config.property_package_args))
        # initial state should not have any flow
        self.initial_state[:].flow_mol.fix(0)

        # add local lists for easy use
        phase_list = self.control_volume.properties_in.phase_list
        pc_set = self.control_volume.properties_in.phase_component_set
        component_list = self.control_volume.properties_in.component_list

        # Get units from property package
        units = self.config.property_package.\
            get_metadata().get_derived_units
        time_units = units("time")
        energy_units = units("energy")
        power_units = units("power")
        if (self.control_volume.properties_in[
                self.flowsheet().config.time.first()]
                .get_material_flow_basis() == MaterialFlowBasis.molar):
            flow_units = units("flow_mole")
        elif (self.control_volume.properties_in[
                self.flowsheet().config.time.first()]
              .get_material_flow_basis() == MaterialFlowBasis.mass):
            flow_units = units("flow_mass")
        else:
            flow_units = None

        # compute gas constant value in the base units
        gas_constant = Ideal.gas_constant(
            self.control_volume.properties_in[0])

        # Add Inlet and Outlet Ports
        self.add_inlet_port()
        self.add_outlet_port()

        # Define Vars for Tank volume calculations
        self.tank_diameter = Var(
            self.flowsheet().config.time,
            within=NonNegativeReals,
            initialize=1.0,
            doc="Diameter of storage tank in m",
            units=pyunits.m)

        self.tank_length = Var(
            self.flowsheet().config.time,
            within=NonNegativeReals,
            initialize=1.0,
            doc="Length of storage tank in m",
            units=pyunits.m)

        # Tank volume calculation
        @self.Constraint(self.flowsheet().config.time)
        def volume_cons(b, t):
            return (b.control_volume.volume[t] ==
                    const.pi * b.tank_length[t] *
                    ((b.tank_diameter[t]/2)**2))

        # define Vars for the model
        self.dt = Var(
            self.flowsheet().config.time,
            domain=NonNegativeReals,
            initialize=100,
            doc="time step for holdup calculation",
            units=time_units)

        self.heat_duty = Var(
            self.flowsheet().config.time,
            domain=Reals,
            initialize=0.0,
            doc="Heat transfered through walls, is 0 for adiabatic",
            units=power_units)

        self.material_accumulation = Var(
            self.flowsheet().config.time,
            pc_set,
            within=Reals,
            initialize=1.0,
            doc="Accumulation of material in tank",
            units=flow_units)

        self.energy_accumulation = Var(
            self.flowsheet().config.time,
            phase_list,
            within=Reals,
            initialize=1.0,
            doc="Energy accumulation",
            units=power_units)

        self.material_holdup = Var(
            self.flowsheet().config.time,
            pc_set,
            within=Reals,
            initialize=1.0,
            doc="Material holdup in tank",
            units=flow_units*time_units)

        self.energy_holdup = Var(
            self.flowsheet().config.time,
            phase_list,
            within=Reals,
            initialize=1.0,
            doc="Energy holdup in tank",
            units=energy_units)

        self.initial_material_holdup = Var(
            self.flowsheet().config.time,
            pc_set,
            within=Reals,
            initialize=1.0,
            doc="Initial material holdup in tank",
            units=flow_units*time_units)

        self.initial_energy_holdup = Var(
            self.flowsheet().config.time,
            phase_list,
            within=Reals,
            initialize=1.0,
            doc="Initial energy holdup in tank",
            units=energy_units)

        # Adiabatic operations are assumed
        # Fixing the heat_duty to 0 here to avoid any misakes at use
        # To Do: remove this once the isothermal constraints are added
        self.heat_duty.fix(0)

        # Computing initial material and energy holdup in the tank 
        # using initial state Pressure and Temperature of the tank
        # Assuming Ideal gas law 
        @self.Constraint(self.flowsheet().config.time,
                         phase_list,
                         doc="Initial material holdup")
        def initial_material_holdup_rule(b, t, p):
            return (
                sum(b.initial_material_holdup[t, p, j]
                    for j in component_list)
                == b.initial_state[t].pressure
                * b.control_volume.volume[t]
                / (b.initial_state[t].temperature * gas_constant)
                )

        @self.Constraint(self.flowsheet().config.time,
                         phase_list,
                         doc="Initial energy holdup")
        def initial_energy_holdup_rule(b, t, p):
            return (
                b.initial_energy_holdup[t, p] ==
                (sum(b.initial_material_holdup[t, p, j]
                     for j in component_list)
                * b.initial_state[t].enth_mol_phase[p])
                )

        # component material balances
        @self.Constraint(self.flowsheet().config.time,
                         pc_set,
                         doc="Material balances")
        def material_balances(b, t, p, j):
            if (p, j) in pc_set:
                return (
                    b.material_accumulation[t, p, j] == 
                    (b.control_volume.properties_in[t].\
                     get_material_flow_terms(p, j) -
                     b.control_volume.properties_out[t].\
                         get_material_flow_terms(p, j))
                    )
            else:
                return Constraint.Skip

        # integration of material accumulation
        @self.Constraint(self.flowsheet().config.time,
                         pc_set,
                         doc="Material holdup integration")
        def material_holdup_integration(b, t, p, j):
            if (p, j) in pc_set:
                return b.material_holdup[t, p, j] == (
                      b.dt[t]
                      * b.material_accumulation[t, p, j]
                      + b.initial_material_holdup[t, p, j])

        # material holdup calculation
        @self.Constraint(self.flowsheet().config.time,
                          pc_set,
                          doc="Material holdup calculations")
        def material_holdup_calculation(b, t, p, j):
            if (p, j) in pc_set:
                return (
                    b.material_holdup[t, p, j] == (
                        b.control_volume.volume[t] *
                        b.control_volume.phase_fraction[t, p] *
                        b.control_volume.properties_out[t].\
                            get_material_density_terms(p, j)))

        # energy balances
        @self.Constraint(self.flowsheet().config.time,
                          doc="Total Enthalpy Balance")
        def enthalpy_balances(b, t):
            return (
                sum(b.energy_accumulation[t, p] for p in phase_list) == (
                    sum(b.control_volume.properties_in[t].\
                        get_enthalpy_flow_terms(p) for p in phase_list)
                    - sum(b.control_volume.properties_out[t].\
                        get_enthalpy_flow_terms(p) for p in phase_list)
                    + b.heat_duty[t]))

        # energy holdup calculation
        @self.Constraint(self.flowsheet().config.time,
                         phase_list,
                         doc="Energy holdup integration")
        def energy_holdup_integration(b, t, p):
            return b.energy_holdup[t, p] == (
                  sum(b.material_holdup[t, p, j]
                      for j in component_list)
                  * b.control_volume.properties_out[t].\
                      enth_mol_phase[p])

        # computing specific heat capacity ratio (cp/cv)
        # NOTE: using ideal gas property Cv = Cp - R
        @self.Expression(self.flowsheet().config.time,
                         doc="specific heat ratio, cp/cv")
        def gama(b, t):
            for p in phase_list:
                if p == 'Vap':
                    return (
                        b.control_volume.properties_in[t].\
                            cp_mol_phase[p]
                        / (b.control_volume.properties_in[t].\
                           cp_mol_phase[p]
                           - gas_constant)
                        )
                else:
                    return 1

        # Add constraint to determine the tank final temperature (T out)
        # This constraint is derived from the 1st law of thermodynamics and 
        # Ideal gas law assuming an adiabatic process
        # Reference: Yang (2009). Int. J. of Hydrogen Energy 34, 6712-6721.

        # TO DO: add pressure out equation for an isothermal process

        @self.Expression(self.flowsheet().config.time,
                         doc="Dimensionles time")
        def dim_less_time(b, t):
            return (
                self.control_volume.properties_in[t].flow_mol
                * b.dt[t]
                / sum(b.initial_material_holdup[t, p, j]
                      for (p, j) in pc_set)
                )

        @self.Constraint(self.flowsheet().config.time,
                         doc="Tank Temperature calculations")
        def tank_temperature_calculation(b, t):
            return b.control_volume.properties_out[t].temperature == (
                  b.gama[t] *
                  b.control_volume.properties_in[t].temperature
                  + ((b.initial_state[t].temperature
                      - b.gama[t] *
                      b.control_volume.properties_in[t].temperature)
                     / (1 + b.dim_less_time[t])))

    def initialize(blk, state_args={}, outlvl=idaeslog.NOTSET,
                   solver='ipopt', optarg={'tol': 1e-6}):
        '''
        compressed hydrogen gas tank initialization routine.

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                           package(s) for the control_volume of the model to
                           provide an initial state for initialization
                           (see documentation of the specific property package)
                           (default = None).
            outlvl : sets output level of initialisation routine

                     * 0 = no output (default)
                     * 1 = return solver state for each step in routine
                     * 2 = return solver state for each step in subroutines
                     * 3 = include solver output infomation (tee=True)

            optarg : solver options dictionary object (default={'tol': 1e-6})
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')

        Returns:
            None
        '''
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        opt = pyo.SolverFactory(solver)
        opt.options = optarg

        init_log.info_low("Starting initialization...")

        flags = blk.control_volume.initialize(state_args=state_args,
                                              outlvl=outlvl,
                                              optarg=optarg,
                                              solver=solver)
        init_log.info_high("Initialization Step 1 Complete.")

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high(
                "Initialization Step 2 {}.".format(idaeslog.condition(res))
            )

        blk.control_volume.release_state(flags, outlvl)
        init_log.info("Initialization Complete.")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        if hasattr(self, "initial_state"):
            for t, v in self.initial_state.items():
                iscale.set_scaling_factor(v.flow_mol, 1e-3)
                iscale.set_scaling_factor(v.pressure, 1e-5)
                iscale.set_scaling_factor(v.temperature, 1e-1)

        if hasattr(self, "tank_diameter"):
            for t, v in self.tank_diameter.items():
                iscale.set_scaling_factor(v, 1)

        if hasattr(self, "tank_length"):
            for t, v in self.tank_length.items():
                iscale.set_scaling_factor(v, 1)

        if hasattr(self, "heat_duty"):
            for t, v in self.heat_duty.items():
                iscale.set_scaling_factor(v, 1e-5)

        if hasattr(self, "material_accumulation"):
            for (t, p, j), v in self.material_accumulation.items():
                iscale.set_scaling_factor(v, 1e-3)

        if hasattr(self, "energy_accumulation"):
            for (t, p), v in self.energy_accumulation.items():
                iscale.set_scaling_factor(v, 1e-3)

        if hasattr(self, "material_holdup"):
            for (t, p, j), v in self.material_holdup.items():
                iscale.set_scaling_factor(v, 1e-5)

        if hasattr(self, "energy_holdup"):
            for (t, p), v in self.energy_holdup.items():
                iscale.set_scaling_factor(v, 1e-5)

        if hasattr(self, "initial_material_holdup"):
            for (t, p, j), v in self.initial_material_holdup.items():
                iscale.set_scaling_factor(v, 1e-5)

        if hasattr(self, "initial_energy_holdup"):
            for (t, p), v in self.initial_energy_holdup.items():
                iscale.set_scaling_factor(v, 1e-5)

        # Volume constraint
        if hasattr(self, "volume_cons"):
            for t, c in self.volume_cons.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.tank_length[t], 
                        default=1, warning=True))

        # Initial Material Holdup Rule
        if hasattr(self, "initial_material_holdup_rule"):
            for (t, i), c in self.initial_material_holdup_rule.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.material_holdup[t, i, j], 
                        default=1, warning=True))

        # Initial Energy Holdup Rule
        if hasattr(self, "initial_energy_holdup_rule"):
            for (t, i), c in self.initial_energy_holdup_rule.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.energy_holdup[t, i], 
                        default=1, warning=True))

        # Material Balances
        if hasattr(self, "material_balances"):
            for (t, i, j), c in self.material_balances.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.material_accumulation[t, i, j], 
                        default=1, warning=True))

        # Material Holdup Integration
        if hasattr(self, "material_holdup_integration"):
            for (t, i, j), c in self.material_holdup_integration.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.material_holdup[t, i, j], 
                        default=1, warning=True))

        # Material Holdup Constraints
        if hasattr(self, "material_holdup_calculation"):
            for (t, i, j), c in self.material_holdup_calculation.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.material_holdup[t, i, j], 
                        default=1, warning=True))

        # Enthalpy Balances
        if hasattr(self, "enthalpy_balances"):
            for t, c in self.enthalpy_balances.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.energy_accumulation[t, p], 
                        default=1, warning=True))

        # Energy Holdup Integration
        if hasattr(self, "energy_holdup_integration"):
            for (t, i), c in self.energy_holdup_integration.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.energy_holdup[t, i], 
                        default=1, warning=True))

        # Tank Temperature Calculation
        if hasattr(self, "tank_temperature_calculation"):
            for t, c in self.tank_temperature_calculation.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(
                        self.initial_state[t].temperature, 
                        default=1, warning=True))
