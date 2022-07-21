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
Ideal property package for H2 vapor
"""
# Import Python libraries
import logging

from pyomo.environ import units as pyunits

# Import IDAES cores
from idaes.core import VaporPhase, Component, PhaseType as PT

from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ideal import Ideal

from idaes.models.properties.modular_properties.pure.NIST import NIST

# Set up logger
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Configuration dictionary for an ideal Benzene-Toluene system

# Data Sources:
# [1] The Properties of Gases and Liquids (1987)
#     4th edition, Chemical Engineering Series - Robert C. Reid
# [2] The NIST Webbook, https://webbook.nist.gov/, retrieved 15th Dec 2020

configuration = {
    # Specifying components
    "components": {
        'hydrogen': {"type": Component,
                     "valid_phase_types": PT.vaporPhase,
                     "elemental_composition": {"H": 2},
                     "cp_mol_ig_comp": NIST,
                     "enth_mol_ig_comp": NIST,
                     "entr_mol_ig_comp": NIST,
                     "parameter_data": {
                         "mw": (2.016e-3, pyunits.kg/pyunits.mol),  # [1]
                         "pressure_crit": (12.9e5, pyunits.Pa),  # [1]
                         "temperature_crit": (33.0, pyunits.K),  # [1]
                         "cp_mol_ig_comp_coeff": {
                             'A': (33.066178,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K),
                             'B': (-11.363417,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kK),
                             'C': (11.432816,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kiloK**2),
                             'D': (-2.772874,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kiloK**3),
                             'E': (-0.158558,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K*pyunits.kiloK**2),
                             'F': (-9.980797,  # [2]
                                   pyunits.kJ/pyunits.mol),
                             'G': (172.707974,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K),
                             'H': (0,  # [2]
                                   pyunits.kJ/pyunits.mol)}}}},

    # Specifying phases
    "phases":  {'Vap': {"type": VaporPhase,
                        "equation_of_state": Ideal}},

    # Set base units of measurement
    "base_units": {"time": pyunits.s,
                   "length": pyunits.m,
                   "mass": pyunits.kg,
                   "amount": pyunits.mol,
                   "temperature": pyunits.K},

    # Specifying state definition
    "state_definition": FTPx,
    "state_bounds": {"flow_mol": (0, 100, 100000, pyunits.mol/pyunits.s),
                     "temperature": (273.15, 300, 1000, pyunits.K),
                     "pressure": (5e4, 1e5, 1e6, pyunits.Pa)},
    "pressure_ref": (101325, pyunits.Pa),  # [2]
    "temperature_ref": (298.15, pyunits.K)}  # [2]
