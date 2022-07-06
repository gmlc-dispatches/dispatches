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
__author__ = "Radhakrishna Tumbalam Gooty"

"""
Basic tests for the simplified hydrogen tank model
"""

import pytest

# Pyomo imports
from pyomo.environ import (ConcreteModel,
                           TerminationCondition)
from pyomo.util.check_units import assert_units_consistent

# IDAES imports
from idaes.core import FlowsheetBlock
from idaes.models.properties.modular_properties.base.generic_property \
    import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.misc import get_solver

# DISPATCHES imports
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration as h2_ideal_config
from dispatches.models.nuclear_case.unit_models.hydrogen_tank_simplified import SimpleHydrogenTank

# Get the default solver for testing
solver = get_solver()


@pytest.fixture()
def build_model():
    # Create the ConcreteModel and the FlowSheetBlock
    m = ConcreteModel(name="H2TankModel")
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # Load thermodynamic package
    m.fs.h2ideal_props = GenericParameterBlock(default=h2_ideal_config)

    # Add hydrogen tank
    m.fs.h2_tank = SimpleHydrogenTank(default={
        "property_package": m.fs.h2ideal_props})

    # Fix the dof of the tank and initialize
    m.fs.h2_tank.inlet.pressure.fix(101325)
    m.fs.h2_tank.inlet.temperature.fix(300)
    m.fs.h2_tank.inlet.flow_mol.fix(25)
    m.fs.h2_tank.inlet.mole_frac_comp[0, "hydrogen"].fix(1)

    m.fs.h2_tank.dt.fix(3600)
    m.fs.h2_tank.tank_holdup_previous.fix(0)
    m.fs.h2_tank.outlet_to_turbine.flow_mol.fix(10)
    m.fs.h2_tank.outlet_to_pipeline.flow_mol.fix(10)
    m.fs.h2_tank.outlet_to_turbine.mole_frac_comp[0, "hydrogen"].fix(1)
    m.fs.h2_tank.outlet_to_pipeline.mole_frac_comp[0, "hydrogen"].fix(1)

    assert degrees_of_freedom(m) == 0

    m.fs.h2_tank.initialize()

    res = solver.solve(m)
    assert res.solver.termination_condition == TerminationCondition.optimal

    return m


@pytest.mark.unit
def test_constraints(build_model):
    m = build_model
    tank = m.fs.h2_tank

    # Constraints
    assert hasattr(tank, "eq_temperature_1")
    assert hasattr(tank, "eq_temperature_2")
    assert hasattr(tank, "eq_pressure_1")
    assert hasattr(tank, "eq_pressure_2")
    assert hasattr(tank, "tank_material_balance")

    # Additional variables
    assert hasattr(tank, "tank_holdup_previous")
    assert hasattr(tank, "tank_holdup")
    assert hasattr(tank, "dt")

    # Inlet and outlet ports
    assert hasattr(tank, "inlet")
    assert hasattr(tank, "outlet_to_turbine")
    assert hasattr(tank, "outlet_to_pipeline")

    assert_units_consistent(m)


@pytest.mark.unit
def test_solution(build_model):
    m = build_model
    tank = m.fs.h2_tank

    assert (tank.outlet_to_pipeline.pressure[0].value ==
            pytest.approx(101325, 1e-2))
    assert (tank.outlet_to_pipeline.temperature[0].value ==
            pytest.approx(300, 1e-2))
    assert (tank.outlet_to_turbine.pressure[0].value ==
            pytest.approx(101325, 1e-2))
    assert (tank.outlet_to_turbine.temperature[0].value ==
            pytest.approx(300, 1e-2))
    assert (tank.tank_holdup[0].value ==
            pytest.approx(3600 * 5, 1e-2))

