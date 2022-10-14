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
# Configuration dictionary for an ideal Air hydrogen system

# Data Sources:
# [1] The Properties of Gases and Liquids (1987)
#     4th edition, Chemical Engineering Series - Robert C. Reid
# [2] The NIST Webbook, https://webbook.nist.gov/, retrieved 15th Dec 2020

configuration = {
    # Specifying components
    "components": {
        'hydrogen': {"type": Component,
                     "elemental_composition": {"H": 2},
                     "valid_phase_types": PT.vaporPhase,
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
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kK**2),
                             'D': (-2.772874,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kK**3),
                             'E': (-0.158558,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K*pyunits.kK**2),
                             'F': (-9.980797,  # [2]
                                   pyunits.J/pyunits.mol),
                             'G': (172.707974,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K),
                             'H': (0,  # [2]
                                   pyunits.kJ/pyunits.mol)}}},
        'nitrogen': {"type": Component,
                     "elemental_composition": {"N": 2},
                     "valid_phase_types": PT.vaporPhase,
                     "cp_mol_ig_comp": NIST,
                     "enth_mol_ig_comp": NIST,
                     "entr_mol_ig_comp": NIST,
                     "parameter_data": {
                         "mw": (28.0134e-3, pyunits.kg / pyunits.mol),  # [1]
                         "pressure_crit": (34.0e5, pyunits.Pa),  # [1]
                         "temperature_crit": (126.15, pyunits.K),  # [1]
                         "cp_mol_ig_comp_coeff": {
                          'A': (19.50583,  # [2]
                                pyunits.J / pyunits.mol / pyunits.K),
                          'B': (19.88705,  # [2]
                                pyunits.J / pyunits.mol / pyunits.K / pyunits.kK),
                          'C': (-8.598535,  # [2]
                                pyunits.J / pyunits.mol / pyunits.K / pyunits.kK ** 2),
                          'D': (1.369784,  # [2]
                                pyunits.J / pyunits.mol / pyunits.K / pyunits.kK ** 3),
                          'E': (0.527601,  # [2]
                                pyunits.J / pyunits.mol / pyunits.K * pyunits.kK ** 2),
                          'F': (-4.935202,  # [2]
                                pyunits.J / pyunits.mol),
                          'G': (212.39000,  # [2]
                                pyunits.J / pyunits.mol / pyunits.K),
                          'H': (0,  # [2]
                                pyunits.kJ / pyunits.mol)}}},
        'oxygen':   {"type": Component,
                     "elemental_composition": {"O": 2},
                     "valid_phase_types": PT.vaporPhase,
                     "cp_mol_ig_comp": NIST,
                     "enth_mol_ig_comp": NIST,
                     "entr_mol_ig_comp": NIST,
                     "parameter_data": {
                         "mw": (31.9988e-3, pyunits.kg/pyunits.mol),  # [1]
                         "pressure_crit": (50.5e5, pyunits.Pa),  # [1]
                         "temperature_crit": (154.55, pyunits.K),  # [1]
                         "cp_mol_ig_comp_coeff": {
                             'A': (31.32234,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K),
                             'B': (-20.23531,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kK),
                             'C': (57.86644,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kK**2),
                             'D': (-36.50624,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K/pyunits.kK**3),
                             'E': (-0.007374,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K*pyunits.kK**2),
                             'F': (-8.903471,  # [2]
                                   pyunits.J/pyunits.mol),
                             'G': (246.7945,  # [2]
                                   pyunits.J/pyunits.mol/pyunits.K),
                             'H': (0,  # [2]
                                   pyunits.kJ/pyunits.mol)}}},

        'water':  {"type": Component,
                   "elemental_composition": {"H": 2, "O": 1},
                   "valid_phase_types": PT.vaporPhase,
                   "cp_mol_ig_comp": NIST,
                   "enth_mol_ig_comp": NIST,
                   "entr_mol_ig_comp": NIST,
                   "parameter_data": {
                     "mw": (18.0153e-3, pyunits.kg / pyunits.mol),  # [1]
                     "pressure_crit": (220.5e5, pyunits.Pa),  # [1]
                     "temperature_crit": (647.15, pyunits.K),  # [1]
                     "cp_mol_ig_comp_coeff": {
                       'A': (30.092,  # [2]
                             pyunits.J / pyunits.mol / pyunits.K),
                       'B': (6.832514,  # [2]
                             pyunits.J / pyunits.mol / pyunits.K / pyunits.kK),
                       'C': (6.793435,  # [2]
                             pyunits.J / pyunits.mol / pyunits.K / pyunits.kK ** 2),
                       'D': (-2.53448,  # [2]
                             pyunits.J / pyunits.mol / pyunits.K / pyunits.kK ** 3),
                       'E': (0.082139,  # [2]
                             pyunits.J / pyunits.mol / pyunits.K * pyunits.kK ** 2),
                       'F': (-250.881,  # [2]
                             pyunits.J / pyunits.mol),
                       'G': (223.3967,  # [2]
                             pyunits.J / pyunits.mol / pyunits.K),
                       'H': (0.0,  # [2] -241.8264
                             pyunits.kJ / pyunits.mol)}}},
        'argon': {"type": Component,
                  "elemental_composition": {"Ar": 1},
                  "valid_phase_types": PT.vaporPhase,
                  "cp_mol_ig_comp": NIST,
                  "enth_mol_ig_comp": NIST,
                  "entr_mol_ig_comp": NIST,
                  "parameter_data": {
                    "mw": (39.948e-3, pyunits.kg / pyunits.mol),  # [1]
                    "pressure_crit": (48.7e5, pyunits.Pa),  # [1]
                    "temperature_crit": (151.15, pyunits.K),  # [1]
                    "cp_mol_ig_comp_coeff": {
                      'A': (20.786,  # [2]
                            pyunits.J / pyunits.mol / pyunits.K),
                      'B': (0.000000282,  # [2]
                            pyunits.J / pyunits.mol / pyunits.K / pyunits.kK),
                      'C': (-0.000000146,  # [2]
                            pyunits.J / pyunits.mol / pyunits.K / pyunits.kK ** 2),
                      'D': (0.00000001092,  # [2]
                            pyunits.J / pyunits.mol / pyunits.K / pyunits.kK ** 3),
                      'E': (-0.0000000366,  # [2]
                            pyunits.J / pyunits.mol / pyunits.K * pyunits.kK ** 2),
                      'F': (-6.19735,  # [2]
                            pyunits.J / pyunits.mol),
                      'G': (179.999,  # [2]
                            pyunits.J / pyunits.mol / pyunits.K),
                      'H': (0.0,  # [2]
                            pyunits.kJ / pyunits.mol)}}}},
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
    "state_bounds": {"flow_mol": (0, 100, 10000, pyunits.mol/pyunits.s),
                     "temperature": (273.15, 300, 2000, pyunits.K),
                     "pressure": (5e4, 1e5, 1e8, pyunits.Pa)},
    "pressure_ref": (101325, pyunits.Pa),  # [2]
    "temperature_ref": (298.15, pyunits.K)}  # [2]
