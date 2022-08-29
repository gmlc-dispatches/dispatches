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
Property package for the hydrogen turbine burning of hydrogen and air to form water
"""

# Import Python libraries
import logging

# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           exp,
                           Expression,
                           Param,
                           PositiveReals,
                           Set,
                           Var,
                           units as pyunits)

# Import IDAES cores
from idaes.core import (declare_process_block_class,
                        MaterialFlowBasis,
                        ReactionParameterBlock,
                        ReactionBlockDataBase,
                        ReactionBlockBase)
from idaes.core.util.misc import add_object_reference

# Set up logger
_log = logging.getLogger(__name__)


@declare_process_block_class("H2ReactionParameterBlock")
class H2ReactionParameterData(ReactionParameterBlock):
    """
    Property Parameter Block Class

    Contains parameters and indexing sets associated with properties for
    superheated steam.

    """
    def build(self):
        '''
        Callable method for Block construction.
        '''
        super(H2ReactionParameterData, self).build()

        self._reaction_block_class = H2ReactionBlock

        # List of valid phases in property package
        self.phase_list = Set(initialize=['Vap'])

        # Component list - a list of component identifiers
        self.component_list = Set(initialize=['argon',
                                              'oxygen',
                                              'nitrogen',
                                              'hydrogen',
                                              'water'])

        # Reaction Index
        self.rate_reaction_idx = Set(initialize=["R1"])

        # Reaction Stoichiometry
        self.rate_reaction_stoichiometry = {("R1", "Vap", "argon"): 0,
                                            ("R1", "Vap", "oxygen"): -1,
                                            ("R1", "Vap", "nitrogen"): 0,
                                            ("R1", "Vap", "hydrogen"): -2,
                                            ("R1", "Vap", "water"): 2,
                                            ("R1", "Liq", "argon"): 0,
                                            ("R1", "Liq", "oxygen"): 0,
                                            ("R1", "Liq", "nitrogen"): 0,
                                            ("R1", "Liq", "hydrogen"): 0,
                                            ("R1", "Liq", "water"): 0}

        # Heat of Reaction
        dh_rxn_dict = {"R1": -4.8366e5}
        self.dh_rxn = Param(self.rate_reaction_idx,
                            initialize=dh_rxn_dict,
                            units=pyunits.J / pyunits.mol,
                            doc="Heat of reaction [J/mol]")

        # Gas Constant
        self.gas_const = Param(within=PositiveReals,
                               mutable=False,
                               default=8.314,
                               doc='Gas Constant [J/mol.K]')

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({
                'k_rxn': {'method': '_rate_constant', 'units': 'm^3/mol.s'},
                'reaction_rate': {'method': "_rxn_rate", 'units': 'mol/m^3.s'}
                })
        obj.add_default_units({'time': pyunits.s,
                               'length': pyunits.m,
                               'mass': pyunits.kg,
                               'amount': pyunits.mol,
                               'temperature': pyunits.K})


class ReactionBlock(ReactionBlockBase):
    """
    This Class contains methods which should be applied to Reaction Blocks as a
    whole, rather than individual elements of indexed Reaction Blocks.
    """
    def initialize(blk, outlvl=0, **kwargs):
        '''
        Initialization routine for reaction package.

        Keyword Arguments:
            outlvl : sets output level of initialization routine

                     * 0 = no output (default)
                     * 1 = report after each step

        Returns:
            None
        '''
        if outlvl > 0:
            _log.info('{} Initialization Complete.'.format(blk.name))


@declare_process_block_class("H2ReactionBlock",
                             block_class=ReactionBlock)
class H2ReactionBlockData(ReactionBlockDataBase):
    """
    An example reaction package for saponification of ethyl acetate
    """

    def build(self):
        """
        Callable method for Block construction
        """
        super(H2ReactionBlockData, self).build()

        # Heat of reaction - no _ref as this is the actual property
        add_object_reference(
                self,
                "dh_rxn",
                self.config.parameters.dh_rxn)

    def get_reaction_rate_basis(b):
        return MaterialFlowBasis.molar
