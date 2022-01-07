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
import pytest
import itertools

from .RE_flowsheet import *


def test_h2_valve_opening():
    opening = 0.0001
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    h2_tank, tank_valve = add_h2_tank(m, 3, 0.3, opening)

    h2_tank.inlet.pressure.fix(8e6)
    h2_tank.inlet.temperature.fix(300)
    h2_tank.inlet.flow_mol[0] = 0
    tank_valve.outlet.pressure.fix(8e6)

