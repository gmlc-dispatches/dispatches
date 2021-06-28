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
Basic tests for Compressed hydrogen gas (chg) tank model
Author: Naresh Susarla.
Date: May 27, 2021
"""

# Import Python libraries
import pytest

# Import Pyomo libraries
from pyomo.environ import (ConcreteModel,
                           SolverStatus,
                           Var,
                           value,
                           TerminationCondition)

# Import IDAES
from idaes.core.util import get_solver
from idaes.core import (FlowsheetBlock,
                        MomentumBalanceType)
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester

# Import unit model and property package
from dispatches.models.nuclear_case.unit_models.\
    hydrogen_tank import HydrogenTank
from dispatches.models.nuclear_case.properties.h2_ideal_vap \
    import configuration

# Get default solver for testing
solver = get_solver()

@pytest.mark.unit
def test_config():
    # Create the ConcreteModel and the FlowSheetBlock
    m = ConcreteModel(name="H2TankModel")
    m.fs = FlowsheetBlock(default={"dynamic": False})
    
    # Add hydrogen property parameter block
    m.fs.properties = GenericParameterBlock(default=configuration)
    
    # Create an instance of the CHG tank
    m.fs.unit = HydrogenTank(default={"property_package": m.fs.properties,
                                      "dynamic": False})
    
    # Check unit config arguments
    assert len(m.fs.unit.config) == 5

    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert m.fs.unit.config.momentum_balance_type == \
        MomentumBalanceType.pressureTotal
    assert m.fs.unit.config.property_package is m.fs.properties

class TestH2IdealVap(object):
    @pytest.fixture(scope="class")
    def hydrogentank(self):
        # Create the ConcreteModel and the FlowSheetBlock
        m = ConcreteModel(name="H2TankModel")
        m.fs = FlowsheetBlock(default={"dynamic": False})
        
        # Add hydrogen property parameter block
        m.fs.properties = GenericParameterBlock(default=configuration)
        
        # Create an instance of the Hydrogen tank
        m.fs.unit = HydrogenTank(default={"property_package": m.fs.properties,
                                          "dynamic": False})

        # Fix tank geometry
        m.fs.unit.tank_diameter.fix(0.1)
        m.fs.unit.tank_length.fix(0.3)
        
        # Fix initial state of tank
        m.fs.unit.previous_state[0].temperature.fix(300)
        m.fs.unit.previous_state[0].pressure.fix(1e5)
        
        # Fix inlet state
        m.fs.unit.control_volume.properties_in[0].flow_mol.fix(100)
        m.fs.unit.control_volume.properties_in[0].mole_frac_comp.fix(1)
        m.fs.unit.control_volume.properties_in[0].temperature.fix(300)
        m.fs.unit.control_volume.properties_in[0].pressure.fix(1.10325e5)
        
        # Fix Duration of Operation (Time Step)
        m.fs.unit.dt[0].fix(100)
        
        # Fix the outlet flow to zero for tank filling type operation
        m.fs.unit.control_volume.properties_out[0].flow_mol.fix(0)
        
        # Setting the bounds on the state variables
        m.fs.unit.control_volume.properties_in[0].pressure.setub(1e15)
        m.fs.unit.control_volume.properties_out[0].pressure.setub(1e15)
        
        return m

    @pytest.mark.unit
    def test_build(self, hydrogentank):
        assert hasattr(hydrogentank.fs.unit, "volume_cons")
        assert hasattr(hydrogentank.fs.unit, "control_volume")
        assert hasattr(hydrogentank.fs.unit, "previous_state")
        assert hasattr(hydrogentank.fs.unit, "material_balances")
        assert hasattr(hydrogentank.fs.unit, "enthalpy_balances")
        assert hasattr(hydrogentank.fs.unit, "material_holdup_calculation")
        assert hasattr(hydrogentank.fs.unit, "tank_temperature_calculation")
        assert isinstance(hydrogentank.fs.unit.tank_diameter, Var)
        assert isinstance(hydrogentank.fs.unit.tank_length, Var)
        assert isinstance(hydrogentank.fs.unit.heat_duty, Var)
        assert isinstance(hydrogentank.fs.unit.previous_material_holdup, Var)
        assert isinstance(hydrogentank.fs.unit.previous_energy_holdup, Var)
    
    # Check degrees of freedom
    @pytest.mark.unit
    def test_dof(self, hydrogentank):
        assert degrees_of_freedom(hydrogentank) == 0
    
    # Check initialization
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, hydrogentank):
        initialization_tester(hydrogentank)

    # Check for optimal solution
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, hydrogentank):
        results = solver.solve(hydrogentank)

        assert results.solver.termination_condition == \
            TerminationCondition.optimal
        assert results.solver.status == SolverStatus.ok

    # Check for solution
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, hydrogentank):
        assert (value(hydrogentank.fs.unit.control_volume.properties_out[0].pressure)
                == pytest.approx(10612813255.63, rel=1e-1))
        assert (value(hydrogentank.fs.unit.control_volume.properties_out[0].temperature)
                == pytest.approx(300.75, rel=1e-1))
        assert (value(hydrogentank.fs.unit.control_volume.properties_out[0].\
                      dens_mol_phase["Vap"])
                == pytest.approx(4244171.91, rel=1e-1))
    
    # Try another inlet temperature
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution2(self, hydrogentank):
        hydrogentank.fs.unit.control_volume.properties_in[0].temperature.fix(400)
        results = solver.solve(hydrogentank)
        assert (results.solver.termination_condition ==
                TerminationCondition.optimal)
        
        assert (value(hydrogentank.fs.unit.control_volume.properties_out[0].pressure)
                == pytest.approx(15536853611.796, rel=1e-1))
        assert (value(hydrogentank.fs.unit.control_volume.properties_out[0].temperature)
                == pytest.approx(440.29, rel=1e-1))
        assert (value(hydrogentank.fs.unit.control_volume.properties_out[0].\
                      dens_mol_phase["Vap"])
                == pytest.approx(4244171.91, rel=1e-1))

    # Check report function
    @pytest.mark.ui
    @pytest.mark.unit
    def test_report(self, hydrogentank):
        hydrogentank.fs.unit.report()
