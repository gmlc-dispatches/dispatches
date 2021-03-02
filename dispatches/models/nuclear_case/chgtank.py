##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Basic IDAES unit model for compressed hydrogen gas tank.
Uses custom material and energy balance equations
Suitable for use with steady state flowsheets indexed with time
(linking constraints over time index using the 
 material_holdup_init and energy_holdup_init variables must be
 written by the user)

"""

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.environ import (Var,
                           Param,
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
import idaes.logger as idaeslog
from idaes.core.util.constants import Constants as const

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

        # add local lists for easy use
        phase_list = self.control_volume.properties_in.phase_list
        pc_set = self.control_volume.properties_in.phase_component_set

        # Get units from property package
        units = self.config.property_package.get_metadata().get_derived_units
        time_units = units("time")
        energy_units = units("energy")
        power_units = units("power")
        if (self.control_volume.properties_in[self.flowsheet().config.time.first()]
                .get_material_flow_basis() == MaterialFlowBasis.molar):
            flow_units = units("flow_mole")
        elif (self.control_volume.properties_in[self.flowsheet().config.time.first()]
              .get_material_flow_basis() == MaterialFlowBasis.mass):
            flow_units = units("flow_mass")
        else:
            flow_units = None

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
        self.dt = Param(
            initialize=1,
            doc="time step for holdup calculation",
            units=time_units)

        self.heat_duty = Var(
            self.flowsheet().config.time,
            domain=Reals,
            initialize=0.0,
            doc="Heat transfered from/to ambient/walls",
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
            doc="Energy in tank",
            units=energy_units)

        self.material_holdup_init = Var(
            self.flowsheet().config.time,
            pc_set,
            within=Reals,
            initialize=1.0,
            doc="Initial amount of material in tank",
            units=flow_units*time_units)

        self.energy_holdup_init = Var(
            self.flowsheet().config.time,
            phase_list,
            within=Reals,
            initialize=1.0,
            doc="Initial energy holdup in tank",
            units=energy_units)

        # material balances and holdup calculations: next 3 constraints
        @self.Constraint(self.flowsheet().config.time,
                         pc_set,
                         doc="Material balances")
        def material_balances(b, t, p, j):
            if (p, j) in pc_set:
                return (
                    b.material_accumulation[t, p, j] == 
                    (b.control_volume.properties_in[t].get_material_flow_terms(p, j) -
                     b.control_volume.properties_out[t].get_material_flow_terms(p, j))
                    )
            else:
                return Constraint.Skip

        @self.Constraint(self.flowsheet().config.time,
                         pc_set,
                         doc="Material holdup integration/aggregation")
        def material_holdup_integration(b, t, p, j):
            if (p, j) in pc_set:
                return b.material_holdup[t, p, j] == (
                      b.dt * b.material_accumulation[t, p, j] +
                      b.material_holdup_init[t, p, j])

        @self.Constraint(self.flowsheet().config.time,
                         pc_set,
                         doc="Material holdup calculations")
        def material_holdup_calculation(b, t, p, j):
            if (p, j) in pc_set:
                return (
                    b.material_holdup[t, p, j] == (
                        b.control_volume.volume[t]*b.control_volume.phase_fraction[t, p] *
                        b.control_volume.properties_out[t].get_material_density_terms(p, j)))

        # energy balances and holdup calculations: next 3 constraints
        @self.Constraint(self.flowsheet().config.time,
                         doc="Energy balances")
        def enthalpy_balances(b, t):
            return b.energy_accumulation[t] == (
                    sum(b.control_volume.properties_in[t].get_enthalpy_flow_terms(p)
                        for p in phase_list) -
                    sum(b.control_volume.properties_out[t].get_enthalpy_flow_terms(p)
                        for p in phase_list))

        @self.Constraint(self.flowsheet().config.time,
                          phase_list,
                         doc="Energy holdup integration/aggregation")
        def energy_holdup_integration(b, t, p):
            return b.energy_holdup[t, p] == (
                  b.dt *
                  b.control_volume.phase_fraction[t, p] *
                  b.energy_accumulation[t] +
                  b.energy_holdup_init[t, p])

        @self.Constraint(self.flowsheet().config.time,
                         phase_list,
                         doc="Energy holdup calculations")
        def energy_holdup_calculation(b, t, p):
            return b.energy_holdup[t, p] == (
                  b.control_volume.volume[t]*b.control_volume.phase_fraction[t, p] *
                  b.control_volume.properties_out[t].get_energy_density_terms(p))

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
        pass