#############################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform to Advance Tightly
# Coupled Hybrid Energy Systems program (DISPATCHES), and is copyright © 2021 by the software owners:
# The Regents of the University of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable Energy, LLC, Battelle
# Energy Alliance, LLC, University of Notre Dame du Lac, et al. All rights reserved.

# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the
# U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted
# for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license
# in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit other to do so.
##############################################################################
"""
Nuclear Flowsheet Tester
Author: Konor Frick
Date: May 11, 2021
"""

import pytest
from pyomo.environ import ConcreteModel, SolverFactory, \
    value

from dispatches.models.nuclear_case.flowsheets.Nuclear_flowsheet import create_model, \
    set_inputs, initialize_model
from idaes.core import FlowsheetBlock


from dispatches.models.nuclear_case.properties.hturbine_ideal_vap \
    import configuration
import dispatches.models.nuclear_case.properties.h2_reaction \
    as reaction_props
from dispatches.models.nuclear_case.unit_models.\
    hydrogen_turbine_unit import HydrogenTurbine
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock

def test_nuclear_fs():
    m = create_model()
    m = set_inputs(m)
    m = initialize_model(m)

    solver = SolverFactory('ipopt')
    res = solver.solve(m, tee=True)



    #print("#### PEM ###")
    #print("Hydrogen flow out of PEM (mol/sec)",
     #     m.fs.pem.outlet.flow_mol[0].value)
    #print("Hydrogen flow out of PEM (kg/sec)", m.fs.H2_production.expr)
    #print("Hydrogen flow out of PEM (kg/hr)", m.fs.H2_production.expr * 3600)

    #print("#### Mixer ###")
    #m.fs.mixer.report()

    #print("#### Hydrogen Turbine ###")
    #m.fs.h2_turbine.compressor.report()
    #m.fs.h2_turbine.stoic_reactor.report()
    #m.fs.h2_turbine.turbine.report()

