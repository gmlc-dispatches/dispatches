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
Property package for Therminol-66
Authored by: Konor Frick and Jaffer Ghouse
Date: 01/20/2021
Source: Therminol 66, High Performance Highly Stable Heat Transfer Fluid (0C to 345C), Solutia.
"""

# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           NonNegativeReals,
                           Param,
                           PositiveReals,
                           Reals,
                           units,
                           value,
                           Var,
                           exp,
                           SolverFactory)

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
import idaes.logger as idaeslog
from idaes.core.solvers import get_solver

# Some more inforation about this module
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

        # Add enthalpy parameters

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({
                'flow_vol': {'method': None, 'units': 'm^3/s'},
                'pressure': {'method': None, 'units': 'Pa'},
                'temperature': {'method': None, 'units': 'K'},
                'conc_mol_comp': {'method': None, 'units': 'mol/m^3'},
                'dens_mol': {'method': None, 'units': 'mol/m^3'}})
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
                   solver=None, optarg=None):
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

        opt = get_solver(solver=solver, options=optarg)

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
    Property package for properties for thermal oil
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(ThermalOilStateBlockData, self).build()

        # Create state variables
        self.make_state_vars()

        # Create required properties
        self.make_properties()

    def make_state_vars(self):
        self.flow_mass = Var(initialize=1.0,
                             domain=NonNegativeReals,
                             doc="Total mass flow [Kg/s]",
                             units=units.kg/units.s)
        self.temperature = Var(initialize=523,
                               domain=NonNegativeReals,
                               doc="Temperature of thermal oil [K]",
                               bounds=(260, 616),
                               units=units.K)
        self.pressure = Var(initialize=101325,
                            domain=NonNegativeReals,
                            doc="Pressure [Pa]",
                            units=units.Pa)

    def make_properties(self):
        self.cp_mass = Var(initialize=100,
                           domain=NonNegativeReals,
                           doc="specific heat capacity [J/Kg/K]"
                           )
        self.therm_cond = Var(initialize=100,
                              domain=NonNegativeReals,
                              doc="thermal conductivity [W/m/K]")
        self.visc_kin = Var(initialize=1,
                            domain=NonNegativeReals,
                            doc="kinematic viscosity [mm2/s]")
        self.density = Var(initialize=1000,
                           domain=NonNegativeReals,
                           doc="density of the thermal oil [Kg/m3]")

        def rule_cp(self):
            return self.cp_mass == 1e3 * \
                (0.003313 * (self.temperature - 273.15) +
                 0.0000008970785 * (self.temperature - 273.15)**2
                 + 1.496005)

        self.eq_cp = Constraint(rule=rule_cp)

        def rule_visc(self):
            return self.visc_kin == \
                exp(586.375 / (self.temperature - 273.15 + 62.5) - 2.2809)

        self.eq_visc = Constraint(rule=rule_visc)

        def rule_therm_cond(self):
            return self.therm_cond == -0.000033 *\
                (self.temperature - 273.15) - 0.00000015 * \
                (self.temperature - 273.15)**2 + 0.118294

        self.eq_therm_cond = Constraint(rule=rule_therm_cond)

        def rule_density(self):
            return self.density == -0.614254 * (self.temperature - 273.15) \
                - 0.000321 * (self.temperature - 273.15) + 1020.62

        self.eq_density = Constraint(rule=rule_density)

    def get_material_flow_terms(self, p, j):
        return self.flow_mass

    def get_enthalpy_flow_terms(self, p):
        return self.flow_mass *1e3* (0.003313*(self.temperature-273.15)**2/2 +
                                     0.0000008970785*(self.temperature-273.15)**3/3 +
                                     1.496005*(self.temperature-273.15))

    def get_material_density_terms(self, p, j):
        return self.density

    def get_energy_density_terms(self, p):
        return self.density*self.cp_mass

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def define_state_vars(self):
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
