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

"""
Thermo-physical property package for Hitec Salt based on empirical correlations
obtained from literatue. Hitec salt is a tertiary salt mixture consisting
of (% wt.): 40% NaNO2, 7% NaNO3, and 53% KNO3
References:
(1) Sohal et al., (2010) Engineering Database of Liquid Salt Thermophysical
and Thermochemical Properties
(2) Change et al., (2015) Energy Procedia, 69, 779 - 789
updated on 05/18/2021
"""


# Import Python libraries
import logging
# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           Param,
                           PositiveReals,
                           Expression,
                           Var,
                           exp,
                           log,
                           units as pyunits,
                           Reals)
from pyomo.opt import TerminationCondition

# Import IDAES
from idaes.core import (declare_process_block_class,
                        StateBlock,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        StateBlockData,
                        PhysicalParameterBlock)
from idaes.core.util.initialization import solve_indexed_blocks
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import fix_state_vars, revert_state_vars
from idaes.core.base.phases import LiquidPhase
from idaes.core.base.components import Component
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog

# Some more inforation about this module
_author_ = "Naresh Susarla"
__version__ = "0.0.1"


# Logger
_log = logging.getLogger(__name__)
init_log = idaeslog.getInitLogger(__name__)
# *****************************************************************************


@declare_process_block_class("HitecsaltParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):
    """
    Property Parameter Block Class.
    Contains parameters associated with properties for Hitec Salt.
    """

    def build(self):
        """Callable method for Block construction."""
        super(PhysicalParameterData, self).build()
        self._state_block_class = HitecsaltStateBlock
        self._make_params()
        self.set_default_scaling('flow_mass', 0.1)
        self.set_default_scaling('temperature', 0.01)
        self.set_default_scaling('pressure', 1e-5)
        self.set_default_scaling('enth_mass', 1e-5)
        self.set_default_scaling('dens_mass', 1e-3)
        self.set_default_scaling('cp_mass', 1e-3)
        self.set_default_scaling('visc_d_phase', 10)
        self.set_default_scaling('therm_cond_phase', 10)
        self.set_default_scaling('enthalpy_flow_terms', 1e-6)

    def _make_params(self):

        # Add Phase objects
        self.Liq = LiquidPhase()
        self.Hitec_Salt = Component()

#        Specific heat capacity at constant pressure (cp) coefficients
#        Cp in J/kg/K
        self.cp_param_1 = Param(initialize=5806,
                                units=pyunits.J/(pyunits.kg*pyunits.K),
                                doc="Coefficient: specific heat expression")
        self.cp_param_2 = Param(initialize=-10.833,
                                units=pyunits.J/(pyunits.kg*(pyunits.K**2)),
                                doc="Coefficient: specific heat expression")
        self.cp_param_3 = Param(initialize=7.2413E-3,
                                units=pyunits.J/(pyunits.kg*(pyunits.K**3)),
                                doc="Coefficient: specific heat expression")

#        Density (rho) coefficients
#        rho in kg/m3
        self.rho_param_1 = Param(initialize=2293.6,
                                 units=pyunits.kg/(pyunits.m**3),
                                 doc="Coefficient: density expression")
        self.rho_param_2 = Param(initialize=-0.7497,
                                 units=pyunits.kg/(pyunits.K*(pyunits.m**3)),
                                 doc="Coefficient: density expression")

#        Dynamic Viscosity (mu) coefficients
#        Mu in Pa.s
        self.mu_param_1 = Param(initialize=-4.343,
                                units=pyunits.Pa*pyunits.s,
                                doc="Coefficient: dynamic viscosity")
        self.mu_param_2 = Param(initialize=-2.0143,
                                units=pyunits.Pa*pyunits.s/pyunits.K,
                                doc="Coefficient: dynamic viscosity")
        self.mu_param_3 = Param(initialize=-5.011,
                                units=pyunits.Pa*pyunits.s/(pyunits.K**2),
                                doc="Coefficient: dynamic viscosity")

#        Thermal conductivity (kappa) coefficients
#        kappa in W/(m.K)
        self.kappa_param_1 = Param(initialize=0.421,
                                   units=pyunits.W/(pyunits.m*pyunits.K),
                                   doc="Coefficient: thermal conductivity")
        self.kappa_param_2 = Param(initialize=-6.53E-4,
                                   units=pyunits.W/(pyunits.m*(pyunits.K**2)),
                                   doc="Coefficient: thermal conductivity")
        self.kappa_param_3 = Param(initialize=-260,
                                   units=pyunits.W/(pyunits.m*(pyunits.K**3)),
                                   doc="Coefficient: thermal conductivity")

#        Thermodynamic reference state
#        ref_temperature in K
        self.ref_temperature = Param(within=PositiveReals,
                                     mutable=True,
                                     default=298.15,
                                     units=pyunits.K,
                                     doc='Reference temperature')

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({'flow_mass': {'method': None, 'units': 'kg/s'},
                            'temperature': {'method': None, 'units': 'K'},
                            'pressure': {'method': None, 'units': 'Pa'},
                            'cp_mass': {'method': None,
                                                 'units': 'J/kg/K'},
                            'dens_mass': {'method': None, 'units': 'kg/m3'},
                            'enth_mass': {'method': None, 'units': 'J/kg'},
                            'visc_d_phase': {'method': None,
                                             'units': 'Pa.s'},
                            'therm_cond_phase': {'method': None,
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
                   outlvl=idaeslog.WARNING,
                   hold_state=False,
                   state_vars_fixed=False,
                   solver=None,
                   optarg=None):
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
            flags = fix_state_vars(blk, state_args)
        for k in blk.keys():
            if degrees_of_freedom(blk[k]) != 0:
                raise Exception("State vars fixed but degrees of freedom "
                                "for state block is not zero during "
                                "initialization.")

        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Solve property correlation
        results = solve_indexed_blocks(opt, [blk])
        if (results.solver.termination_condition ==
                TerminationCondition.optimal):
            init_log.info('{} Initialisation Step 1 Complete.'.
                          format(blk.name))
        else:
            init_log.warning('{} Initialisation Step 1 Failed.'.
                             format(blk.name))

        init_log.info('Initialization Step 1 Complete.')

        # ---------------------------------------------------------------------
        if state_vars_fixed is False:
            # release state vars fixed during initialization if control
            # volume didn't fix the state vars
            if hold_state is True:
                return flags
            else:
                blk.release_state(flags)
        init_log.info('Initialization Complete.')

    def release_state(blk, flags, outlvl=idaeslog.NOTSET):
        # Method to release states only if explicitly called

        if flags is None:
            return
        # Unfix state variables
        revert_state_vars(blk, flags)
        init_log.info('State Released.')


@declare_process_block_class("HitecsaltStateBlock", block_class=_StateBlock)
class HitecsaltStateBlockData(StateBlockData):
    """An example property package for Hitec Salt properties."""

    def build(self):
        """Callable method for Block construction."""
        super(HitecsaltStateBlockData, self).build()
        self._make_state_vars()
        self._make_prop_vars()
        self._make_constraints()

    def _make_state_vars(self):
        """Declare the necessary state variable objects."""
        self.flow_mass = Var(domain=Reals,
                             initialize=100,
                             units=pyunits.kg/pyunits.s,
                             doc='Fluid mass flowrate')
        self.pressure = Var(domain=Reals,
                            initialize=1.01325E5,
                            units=pyunits.Pa,
                            doc='State pressure')
        self.temperature = Var(domain=Reals,
                               initialize=550,
                               units=pyunits.K,
                               doc='State temperature',
                               bounds=(435.15, 788.15))  # i.e., 162 - 515 C

    def _make_prop_vars(self):
        """Make additional variables for calcuations."""

        self.enth_mass = Var(self.phase_list,
                                 initialize=1,
                                 units=pyunits.J/pyunits.kg,
                                 doc='Specific Enthalpy')

    def _make_constraints(self):
        """Create property constraints."""

        # Specific heat capacity
        self.cp_mass = Expression(
            self.phase_list,
            expr=(self.params.cp_param_1 +
                  (self.params.cp_param_2 * self.temperature) +
                  (self.params.cp_param_3 * (self.temperature**2))),
            doc="Specific heat capacity")

        # Density
        self.dens_mass = Expression(
            self.phase_list,
            expr=(self.params.rho_param_1 +
                  (self.params.rho_param_2 * self.temperature)),
            doc="density")

        # Specific Enthalpy
        def enthalpy_correlation(self, p):
            return (
                self.enth_mass[p] ==
                ((self.params.cp_param_1 * self.temperature) +
                 (self.params.cp_param_2 * self.temperature**2) +
                 (self.params.cp_param_3 * self.temperature**3)))
        self.enthalpy_eq = Constraint(self.phase_list,
                                      rule=enthalpy_correlation)

#           Dynamic viscosity and thermal conductivity
#           Ref: (2015) Change et al., Energy Procedia, 69, 779 - 789

        # Dynamic viscosity
        self.visc_d_phase = Expression(
            self.phase_list,
            expr=exp(self.params.mu_param_1 +
                     self.params.mu_param_2 * (log(self.temperature) +
                                               self.params.mu_param_3)),
            doc="dynamic viscosity")

        # Thermal conductivity
        self.therm_cond_phase = Expression(
            self.phase_list,
            expr=self.params.kappa_param_1 +
            self.params.kappa_param_2 * (self.temperature +
                                         self.params.kappa_param_3),
            doc="dynamic viscosity")

        # Enthalpy flow terms
        def rule_enthalpy_flow_terms(b, p):
            return (self.enth_mass[p] * self.flow_mass)

        self.enthalpy_flow_terms = Expression(
            self.config.parameters.phase_list,
            rule=rule_enthalpy_flow_terms)

    def get_material_flow_terms(b, p, j):
        """Define material flow terms for control volume."""
        return b.flow_mass

    def get_enthalpy_flow_terms(b, p):
        """Define enthalpy flow terms for control volume."""
        return b.enthalpy_flow_terms[p]

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def define_state_vars(b):
        """Define state variables for ports."""
        return {"flow_mass": b.flow_mass,
                "temperature": b.temperature,
                "pressure": b.pressure}
