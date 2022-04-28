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
Property package for Therminol-66
Authored by: Konor Frick and Jaffer Ghouse
Edited by: Naresh Susarla and Soraya Rawlings
Date: 12/02/2021
Source: Therminol 66, High Performance Highly Stable Heat Transfer Fluid (0C to 345C), Solutia.
"""

# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           NonNegativeReals,
                           Param,
                           PositiveReals,
                           Expression,
                           Reals,
                           units,
                           value,
                           Var,
                           exp
)
from pyomo.opt import TerminationCondition

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        MaterialFlowBasis,
                        PhysicalParameterBlock,
                        StateBlockData,
                        StateBlock,
                        MaterialBalanceType,
                        EnergyBalanceType)
from idaes.core.phases import LiquidPhase
from idaes.core.components import Component
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import solve_indexed_blocks, \
    fix_state_vars, revert_state_vars
from idaes.core.util import get_solver
import idaes.logger as idaeslog

# Some more information about this module
__author__ = "Jaffer Ghouse, Konor Frick"


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ThermalOilParameterBlock")
class PhysicalParameterData(PhysicalParameterBlock):
    """
    Property Parameter Block Class

    """

    def build(self):
        '''
        Callable method for Block construction.
        '''
        super(PhysicalParameterData, self).build()
        self._state_block_class = ThermalOilStateBlock
        # Add Phase objects
        self.Liq = LiquidPhase()
        # Add Component objects
        self.therminol66 = Component()
        # Add scaling factors
        self.set_default_scaling('flow_mass', 0.1)
        self.set_default_scaling('temperature', 0.01)
        self.set_default_scaling('pressure', 1e-5)
        self.set_default_scaling('enthalpy_mass', 1e-5)
        self.set_default_scaling('density', 1e-3)
        self.set_default_scaling('cp_mass', 1e-3)
        self.set_default_scaling('visc_kin', 10)
        self.set_default_scaling('therm_cond', 10)
        self.set_default_scaling('enthalpy_flow_terms', 1e-6)

        # Add enthalpy parameters

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({'flow_vol': {'method': None, 'units': 'm^3/s'},
                            'temperature': {'method': None, 'units': 'K'},
                            'pressure': {'method': None, 'units': 'Pa'},
                            'conc_mol_comp': {'method': None, 'units': 'mol/m^3'},
                            'dens_mol': {'method': None, 'units': 'mol/m^3'},
                            'cp_mass': {'method': None, 'units': 'J/kg/K'},
                            'enthalpy': {'method': None, 'units': 'J/kg'},
                            'density': {'method': None, 'units': 'kg/m3'},
                            'visc_kin': {'method': None, 'units': 'mm2/s'},
                            'therm_cond': {'method': None, 'units': 'W/m/K'}})
        obj.add_default_units({'time': units.s,
                               'length': units.m,
                               'mass': units.kg,
                               'amount': units.mol,
                               'temperature': units.K})


class _StateBlock(StateBlock):
    """
    This Class contains methods which should be applied to Property Blocks as a
    whole, rather than individual elements of indexed Property Blocks.
    """
    def initialize(self, state_args={}, state_vars_fixed=False,
                   hold_state=False, outlvl=idaeslog.NOTSET,
                   temperature_bounds=(260, 616),
                   solver='ipopt', optarg={'tol': 1e-8}):
        '''
        Initialization routine for property package.

        Keyword Arguments:
        state_args : Dictionary with initial guesses for the state vars
                     chosen. Note that if this method is triggered
                     through the control volume, and if initial guesses
                     were not provied at the unit model level, the
                     control volume passes the inlet values as initial
                     guess.The keys for the state_args dictionary are:

                     flow_mol_comp : value at which to initialize component
                                     flows (default=None)
                     pressure : value at which to initialize pressure
                                (default=None)
                     temperature : value at which to initialize temperature
                                  (default=None)
            outlvl : sets output level of initialization routine
            state_vars_fixed: Flag to denote if state vars have already been
                              fixed.
                              - True - states have already been fixed and
                                       initialization does not need to worry
                                       about fixing and unfixing variables.
                             - False - states have not been fixed. The state
                                       block will deal with fixing/unfixing.
            optarg : solver options dictionary object (default=None)
            solver : str indicating whcih solver to use during
                     initialization (default = 'ipopt')
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
        '''
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl,
                                            tag="properties")

        if state_vars_fixed is False:
            # Fix state variables if not already fixed
            flags = fix_state_vars(self, state_args)

        else:
            # Check when the state vars are fixed already result in dof 0
            for k in self.keys():
                if degrees_of_freedom(self[k]) != 0:
                    raise Exception("State vars fixed but degrees of freedom "
                                    "for state block is not zero during "
                                    "initialization.")

        if optarg is None:
            sopt = {"tol": 1e-8}
        else:
            sopt = optarg

        opt = get_solver(solver, optarg)

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solve_indexed_blocks(opt, [self], tee=slc.tee)
        init_log.info("Initialization Step 1 {}.".
                      format(idaeslog.condition(res)))

        if state_vars_fixed is False:
            if hold_state is True:
                return flags
            else:
                self.release_state(flags)

        init_log.info('Initialization Complete.')

    def release_state(self, flags, outlvl=idaeslog.NOTSET):
        '''
        Method to relase state variables fixed during initialization.

        Keyword Arguments:
            flags : dict containing information of which state variables
                    were fixed during initialization, and should now be
                    unfixed. This dict is returned by initialize if
                    hold_state=True.
            outlvl : sets output level of of logging
        '''
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="properties")

        if flags is None:
            return
        # Unfix state variables
        revert_state_vars(self, flags)
        init_log.info('State Released.')


@declare_process_block_class("ThermalOilStateBlock",
                             block_class=_StateBlock)
class ThermalOilStateBlockData(StateBlockData):
    """
    Property package for Thermal oil properties
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(ThermalOilStateBlockData, self).build()

        # Create state variables
        self._make_state_vars()

        # Create required properties
        self._make_properties()

        # Create constraints
        self._make_constraints()

    def _make_state_vars(self):
        """Declare the necessary state variable objects."""
        self.flow_mass = Var(initialize=1.0,
                             domain=Reals,
                             doc="Total mass flow [Kg/s]",
                             units=units.kg/units.s)
        self.temperature = Var(initialize=523,
                               domain=Reals,
                               doc="Temperature of thermal oil [K]",
                               bounds=(260, 616),
                               units=units.K)
        self.pressure = Var(initialize=101325,
                            domain=Reals,
                            doc="Pressure [Pa]",
                            units=units.Pa)

    def _make_properties(self):
        """Make additional variables for calculations."""

        self.enthalpy_mass = Var(self.phase_list,
                                 initialize=1,
                                 units=units.J/units.kg,
                                 doc='Specific Enthalpy')

    def _make_constraints(self):
        """Create property constraints."""

        # Specific heat capacity
        self.cp_mass = Expression(
            self.phase_list,
            expr=(1e3 * (0.003313 * (self.temperature - 273.15) +
                         0.0000008970785 * (self.temperature - 273.15)**2 +
                         1.496005)),
            doc="Specific heat capacity"
        )

        # Specific Enthalpy
        def enthalpy_correlation(self, p):
            return (
                self.enthalpy_mass[p]
                == (1e3 * (0.003313*(self.temperature-273.15)**2/2 +
                           0.0000008970785*(self.temperature-273.15)**3/3 +
                           1.496005*(self.temperature-273.15))))
        self.enthalpy_eq = Constraint(self.phase_list,
                                      rule=enthalpy_correlation)

        # Viscosity
        self.visc_kin = Expression(
            self.phase_list,
            expr=exp(586.375 / (self.temperature - 273.15 + 62.5) - 2.2809),
            doc=""
        )

        # Thermal conductivity
        self.therm_cond = Expression(
            self.phase_list,
            expr=(-0.000033 *\
                  (self.temperature - 273.15) - 0.00000015 * \
                  (self.temperature - 273.15)**2 + 0.118294),
            doc="Thermal conductivity"
        )

        # Density
        self.density = Expression(
            self.phase_list,
            expr=(-0.614254 * (self.temperature - 273.15) \
                  - 0.000321 * (self.temperature - 273.15) + 1020.62),
            doc="Density"
        )

        # Enthalpy flow terms
        def rule_enthalpy_flow_terms(b, p):
            return (self.enthalpy_mass[p] * self.flow_mass)

        self.enthalpy_flow_terms = Expression(
            self.config.parameters.phase_list,
            rule=rule_enthalpy_flow_terms)

    def get_material_flow_terms(self, p, j):
        return self.flow_mass

    def get_enthalpy_flow_terms(self, p):
        return self.enthalpy_flow_terms[p]

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def define_state_vars(self):
        """Define state variables for ports."""
        return {"flow_mass": self.flow_mass,
                "temperature": self.temperature,
                "pressure": self.pressure}

    def define_display_vars(self):
        return {"flow_mass": self.flow_mass,
                "temperature": self.temperature,
                "pressure": self.pressure}

    def get_material_flow_basis(self):
        return MaterialFlowBasis.mass

    def model_check(self):
        """
        Model checks for property block
        """
        # Check temperature bounds
        if value(self.temperature) < self.temperature.lb:
            _log.error('{} Temperature set below lower bound.'
                       .format(self.name))
        if value(self.temperature) > self.temperature.ub:
            _log.error('{} Temperature set above upper bound.'
                       .format(self.name))

        # Check pressure bounds
        if value(self.pressure) < self.pressure.lb:
            _log.error('{} Pressure set below lower bound.'.format(self.name))
        if value(self.pressure) > self.pressure.ub:
            _log.error('{} Pressure set above upper bound.'.format(self.name))
