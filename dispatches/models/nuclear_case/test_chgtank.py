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
Date: APRIL 17, 2021
"""

# Import Python libraries
import pytest

# Import Pyomo libraries
from pyomo.environ import (ConcreteModel,
                           SolverFactory,
                           Var,
                           value,
                           TerminationCondition)

# Import IDAES
from idaes.core import FlowsheetBlock
from dispatches.models.nuclear_case.chgtank import CHGTank
from dispatches.models.nuclear_case.h2_ideal_vap import configuration
from idaes.generic_models.properties.core.generic.generic_property \
    import GenericParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom

def test_chgtank():
    # Create the ConcreteModel and the FlowSheetBlock
    m = ConcreteModel(name="H2TankModel")
    m.fs = FlowsheetBlock(default={"dynamic": False})
    
    # Add hydrogen property parameter block
    m.fs.properties = GenericParameterBlock(default=configuration)
    
    # Create an instance of the CHG tank
    m.fs.unit = CHGTank(default={"property_package": m.fs.properties,
                                 "dynamic": False,
                                 "has_pressure_change": True})
    
    assert hasattr(m.fs.unit, "volume_cons")
    assert hasattr(m.fs.unit, "control_volume")
    assert hasattr(m.fs.unit, "initial_state")
    assert hasattr(m.fs.unit, "material_balances")
    assert hasattr(m.fs.unit, "enthalpy_balances")
    assert hasattr(m.fs.unit, "material_holdup_calculation")
    assert hasattr(m.fs.unit, "tank_temperature_calculation")
    assert isinstance(m.fs.unit.tank_diameter, Var)
    assert isinstance(m.fs.unit.tank_length, Var)
    assert isinstance(m.fs.unit.heat_duty, Var)
    assert isinstance(m.fs.unit.initial_material_holdup, Var)
    assert isinstance(m.fs.unit.initial_energy_holdup, Var)
    
    
    # Fix tank geometry
    m.fs.unit.tank_diameter.fix(0.1)
    m.fs.unit.tank_length.fix(0.3)
    
    # Fix initial state of tank
    m.fs.unit.initial_state[0].temperature.fix(300)
    m.fs.unit.initial_state[0].pressure.fix(1e5)
    
    # Fix inlet state
    m.fs.unit.control_volume.properties_in[0].flow_mol.fix(100)
    m.fs.unit.control_volume.properties_in[0].mole_frac_comp.fix(1)
    m.fs.unit.control_volume.properties_in[0].temperature.fix(300)
    m.fs.unit.control_volume.properties_in[0].pressure.fix(1.10325e5)
    
    # Fix Duration of Operation (Time Step)
    m.fs.unit.dt[0].fix(100)
    
    # Fix the outlet flow to zero for tank filling type operation
    m.fs.unit.control_volume.properties_out[0].flow_mol.fix(0)
    
    assert degrees_of_freedom(m) == 0
    
    # Initialize unit
    m.fs.unit.initialize()
    
    solver = SolverFactory("ipopt")
    solver.options = {
        "tol": 1e-8,
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }
    
    results = solver.solve(m, tee=True)
    
    assert (results.solver.termination_condition ==
            TerminationCondition.optimal)
    
    assert (value(m.fs.unit.control_volume.properties_out[0].pressure)
            == pytest.approx(14872710319.67, rel=1e-1))
    assert (value(m.fs.unit.control_volume.properties_out[0].temperature)
            == pytest.approx(421.47, rel=1e-1))
    assert (value(m.fs.unit.control_volume.properties_out[0].\
                  dens_mol_phase["Vap"])
            == pytest.approx(4244171.91, rel=1e-1))
    
    # Try another inlet temperature
    m.fs.unit.control_volume.properties_in[0].temperature.fix(400)
    
    results = solver.solve(m, tee=True)
    
    assert (results.solver.termination_condition ==
            TerminationCondition.optimal)
    
    assert (value(m.fs.unit.control_volume.properties_out[0].pressure)
            == pytest.approx(19739284390.23, rel=1e-1))
    assert (value(m.fs.unit.control_volume.properties_out[0].temperature)
            == pytest.approx(559.38, rel=1e-1))
    assert (value(m.fs.unit.control_volume.properties_out[0].\
                  dens_mol_phase["Vap"])
            == pytest.approx(4244171.91, rel=1e-1))
