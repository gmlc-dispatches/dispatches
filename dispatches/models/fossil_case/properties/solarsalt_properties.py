##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
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
Created on Thu May 23 14:11:22 2019
@author: naresh susarla
Property package for Molten Salt for thermal energy storage.
In this version 0.0.1, Molten Salt refers to the solar salt which is
a mixture of NaNO3 and KNO3
Composition of this salt mixture by wt.: 60% NaNO3 + 40% KNO3
property equations source: RELAP5 code by Idaho National Laboratory
obtained from:
2008-Int. J Ther. Sci.-Ferri et al.
Molten salt mixture properties in RELAP5 code for thermodynamic
solar applications (47) 1676 - 1687
"""
# Chages the divide behavior to not do integer division
# from _future_ import division
# Import Python libraries
import logging

# Import Pyomo libraries
from pyomo.environ import Constraint, Param, PositiveReals, Expression, \
        RangeSet, Var, Reals
from pyomo.environ import units as pyunits
from pyomo.opt import TerminationCondition

# Import IDAES
from idaes.core import declare_process_block_class, StateBlock, \
                       MaterialBalanceType, EnergyBalanceType, \
                       StateBlockData, PhysicalParameterBlock
from idaes.core.util.misc import add_object_reference
from idaes.core.util.initialization import solve_indexed_blocks
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.phases import LiquidPhase
from idaes.core.components import Component
from idaes.core.util import get_solver
import idaes.logger as idaeslog

# Some more inforation about this module
_author_ = "Naresh Susarla"
__version__ = "0.0.1"

# Logger
_log = logging.getLogger(__name__)

# **** Requires Temperature in K
# Ref: (2015) Chang et al, Energy Procedia 69, 779 - 789
# Specific Heat Capacity as a function of Temperature, J/kg/K
# def specific_heat_cp(model):
#    return model.specific_heat_cp
#       == 1443 + 0.172 * model.temperature
#
# Thermal Conductivity as a function of Temperature, W/m/K
# def thermal_conductivity(model):
#    return model.thermal_conductivity
#       == 0.443 + 0.00019 * model.temperature
#
# ****************************************************************************


@declare_process_block_class("SolarsaltParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):
    """
    Property Parameter Block Class.
    Contains parameters associated with properties for Solar Salt.
    """

    def build(self):
        """Callable method for Block construction."""
        super(PhysicalParameterData, self).build()
        self._state_block_class = SolarsaltStateBlock
        self._make_params()
        self.set_default_scaling('flow_mass', 0.1)
        self.set_default_scaling('temperature', 0.01)
        self.set_default_scaling('pressure', 1e-5)
        self.set_default_scaling('enthalpy_mass', 1e-5)
        self.set_default_scaling('density', 1e-3)
        self.set_default_scaling('cp_specific_heat', 1e-3)
        self.set_default_scaling('dynamic_viscosity', 10)
        self.set_default_scaling('thermal_conductivity', 10)
        self.set_default_scaling('enthalpy_flow_terms', 1e-6)

    def _make_params(self):
        # Add Phase objects
        self.Liq = LiquidPhase()
        self.Solar_Salt = Component()

#        Specific heat capacity at constant pressure (cp) coefficients
#        Cp in J/kg/K
        cp_param = {1: 1443,
                    2: 0.172}
        self.cp_param = Param(RangeSet(2), initialize=cp_param,
                              doc="specific heat parameters")

#        Density (rho) coefficients
#        rho in kg/m3
        rho_param = {1: 2090,
                     2: -0.636}
        self.rho_param = Param(RangeSet(2), initialize=rho_param,
                               doc="density parameters")

#        Dynamic Viscosity (mu) coefficients
#        Mu in Pa.s
        mu_param = {1: 2.2714E-2,
                    2: -1.2E-4,
                    3: 2.281E-7,
                    4: -1.474E-10}
        self.mu_param = Param(RangeSet(4), initialize=mu_param,
                              doc="dynamic viscosity parameters")

#        Thermal conductivity (kappa) coefficients
#        kappa in W/(m.K)
        kappa_param = {1: 0.443,
                       2: 1.9E-4}
        self.kappa_param = Param(RangeSet(2), initialize=kappa_param,
                                 doc="thermal conductivity parameters")

#        Thermodynamic reference state
#        ref_temperature in K
        self.ref_temperature = Param(within=PositiveReals,
                                     mutable=True,
                                     default=298.15,
                                     doc='Reference temperature [K]')

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({'flow_mass': {'method': None, 'units': 'kg/s'},
                            'temperature': {'method': None, 'units': 'K'},
                            'pressure': {'method': None, 'units': 'Pa'},
                            'cp_specific_heat': {'method': None,
                                                 'units': 'J/kg/K'},
                            'enthalpy': {'method': None, 'units': 'J/kg'},
                            'density': {'method': None, 'units': 'kg/m3'},
                            'dynamic_viscosity': {'method': None,
                                                  'units': 'Pa.s'},
                            'thermal_conductivity': {'method': None,
                                                     'units': 'W/m/K'}})
        obj.add_default_units({'time': pyunits.s,
                               'length': pyunits.m,
                               'mass': pyunits.kg,
                               'amount': pyunits.mol,
                               'temperature': pyunits.K})


class _StateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a
    whole, rather than individual elements of indexed Property Blocks.
    """
    def initialize(blk, state_args=None,
                   outlvl=idaeslog.NOTSET,
                   hold_state=False,
                   state_vars_fixed=False,
                   solver=None,
                   optarg={}):
        """
        Declare initialisation routine.
        Keyword Arguments:
            state_args = to be used if state block initialized independent of
                         control volume initialize
            outlvl : sets output level of initialisation routine
                     * 0 = no output (default)
                     * 1 = return solver state for each step in routine
                     * 2 = include solver output infomation (tee=True)
            optarg : solver options dictionary object (default=None)
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed by the
                                       control volume 1D. Control volume 0D
                                       does not fix the state vars, so will
                                       be False if this state block is used
                                       with 0D blocks.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            hold_state : flag indicating whether the initialization routine
                         should unfix any state variables fixed during
                         initialization (default=False).
                         - True - states varaibles are not unfixed, and
                                 a dict of returned containing flags for
                                 which states were fixed during
                                 initialization.
                        - False - state variables are unfixed after
                                 initialization by calling the
                                 relase_state method
        Returns:
            If hold_states is True, returns a dict containing flags for
            which states were fixed during initialization.
        """
        # Fix state variables if not already fixed by the control volume block
        if state_vars_fixed is False:
            # Fix state variables if not already fixed
            fflag = {}
            pflag = {}
            tflag = {}

            for k in blk.keys():
                if blk[k].flow_mass.fixed is True:
                    fflag[k] = True
                else:
                    fflag[k] = False
                    if state_args is None:
                        blk[k].flow_mass.fix()
                    else:
                        blk[k].flow_mass.fix(state_args["flow_mass"])

                if blk[k].pressure.fixed is True:
                    pflag[k] = True
                else:
                    pflag[k] = False
                    if state_args is None:
                        blk[k].pressure.fix()
                    else:
                        blk[k].pressure.fix(state_args["pressure"])

                if blk[k].temperature.fixed is True:
                    tflag[k] = True
                else:
                    tflag[k] = False
                    if state_args is None:
                        blk[k].temperature.fix()
                    else:
                        blk[k].temperature.fix(state_args["temperature"])

            flags = {"fflag": fflag, "pflag": pflag, "tflag": tflag}
        else:
            # Check when the state vars are fixed already result in dof 0
            for k in blk.keys():
                if degrees_of_freedom(blk[k]) != 0:
                    raise Exception("State vars fixed but degrees of freedom "
                                    "for state block is not zero during "
                                    "initialization.")
        # Set solver options
        if outlvl > 1:
            stee = True
        else:
            stee = False
        opt = get_solver(solver, optarg)
        # ---------------------------------------------------------------------
        # Solve property correlation
        results = solve_indexed_blocks(opt, [blk], tee=stee)
        if outlvl > 0:
            if results.solver.termination_condition \
                    == TerminationCondition.optimal:
                _log.info('{} Initialisation Step 1 Complete.'
                          .format(blk.name))
            else:
                _log.warning('{} Initialisation Step 1 Failed.'
                             .format(blk.name))
        # ---------------------------------------------------------------------
        # If input block, return flags, else release state
        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                blk.release_state(flags)

        if outlvl > 0:
            if outlvl > 0:
                _log.info('{} Initialisation Complete.'.format(blk.name))

    def release_state(blk, flags, outlvl=0):
        # Method to release states only if explicitly called
        if flags is None:
            return
        # Unfix state variables
        for k in blk.keys():
            if flags['fflag'][k] is False:
                blk[k].flow_mass.unfix()
            if flags['pflag'][k] is False:
                blk[k].pressure.unfix()
            if flags['tflag'][k] is False:
                blk[k].temperature.unfix()

        if outlvl > 0:
            if outlvl > 0:
                _log.info('{} State Released.'.format(blk.name))


@declare_process_block_class("SolarsaltStateBlock", block_class=_StateBlock)
class StateTestBlockData(StateBlockData):
    """An example property package for Molten Salt properties."""

    def build(self):
        """Callable method for Block construction."""
        super(StateTestBlockData, self).build()
        self._make_params()
        self._make_state_vars()
        self._make_prop_vars()
        self._make_constraints()

    def _make_params(self):
        """Make references to the necessary parameters contained."""
        # Thermodynamic reference state
        add_object_reference(self, "ref_temperature",
                             self.config.parameters.ref_temperature)
        # Density Coefficients
        add_object_reference(self, "rho_param",
                             self.config.parameters.rho_param)
        # Specific Enthalpy Coefficients
        add_object_reference(self, "cp_param",
                             self.config.parameters.cp_param)
        # Dynamic Viscosity Coefficients
        add_object_reference(self, "mu_param",
                             self.config.parameters.mu_param)
        # Thermal Conductivity Coefficients
        add_object_reference(self, "kappa_param",
                             self.config.parameters.kappa_param)

    def _make_state_vars(self):
        """Declare the necessary state variable objects."""
        self.flow_mass = Var(domain=Reals,
                             initialize=0.5,
                             units=pyunits.kg/pyunits.s,
                             bounds=(0, None),
                             doc='Fluid mass flowrate [kg/s]')
        self.pressure = Var(domain=Reals,
                            initialize=1.01325E5,
                            units=pyunits.Pa,
                            doc='State pressure [Pa]')
        self.temperature = Var(domain=Reals,
                               initialize=550,
                               units=pyunits.K,
                               doc='State temperature [K]',
                               bounds=(513.15, 853.15))  # 240 - 580 C

    def _make_prop_vars(self):
        """Make additional variables for calcuations."""

        self.enthalpy_mass = Var(self.phase_list,
                                 initialize=1,
                                 doc='Specific Enthalpy [J/kg]')

    def _make_constraints(self):
        """Create property constraints."""

        # temp_in_c = pyunits.convert(
        #      self.temperature,
        #      to_units=pyunits.c)

        # Specific heat capacity
        self.cp_specific_heat = Expression(
            self.phase_list,
            expr=(self.cp_param[1]
                  + (self.cp_param[2] * (self.temperature-273.15))),
            doc="Specific heat capacity")

        # Density (using T in C for the expression, D in kg/m3)
        self.density = Expression(
            self.phase_list,
            expr=self.rho_param[1] + self.rho_param[2]
            * (self.temperature - 273.15),
            doc="density")

        # Specific Enthalpy
        def enthalpy_correlation(self, p):
            return (
                self.enthalpy_mass[p]
                == ((self.cp_param[1] * (self.temperature-273.15))
                    + (self.cp_param[2] * 0.5
                       * (self.temperature-273.15)**2)))
        self.enthalpy_eq = Constraint(self.phase_list,
                                      rule=enthalpy_correlation)

        # Dynamic viscosity
        self.dynamic_viscosity = Expression(
            self.phase_list,
            expr=(self.mu_param[1]
                  + self.mu_param[2] * (self.temperature-273.15)
                  + self.mu_param[3] * (self.temperature-273.15)**2
                  + self.mu_param[4] * (self.temperature-273.15)**3),
            doc="dynamic viscosity")

        # Thermal conductivity
        self.thermal_conductivity = Expression(
            self.phase_list,
            expr=(self.kappa_param[1]
                  + self.kappa_param[2] * (self.temperature-273.15)),
            doc="thermal conductivity")

        # Enthalpy flow terms
        def rule_enthalpy_flow_terms(b, p):
            return (self.enthalpy_mass[p] * self.flow_mass)

        self.enthalpy_flow_terms = Expression(
            self.config.parameters.phase_list,
            rule=rule_enthalpy_flow_terms)

    def get_material_flow_terms(b, p, j):
        """Define material flow terms for control volume."""
        return b.flow_mass

    def get_enthalpy_flow_terms(self, p):
        """Define enthalpy flow terms for control volume."""
        return self.enthalpy_flow_terms[p]

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def define_state_vars(b):
        """Define state variables for ports."""
        return {"flow_mass": b.flow_mass,
                "temperature": b.temperature,
                "pressure": b.pressure}

