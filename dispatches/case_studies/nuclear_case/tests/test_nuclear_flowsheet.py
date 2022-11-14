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
Nuclear Flowsheet Tester
Author: Konor Frick
Date: May 11, 2021
"""

import pytest
from pyomo.environ import (ConcreteModel,
                           TerminationCondition,
                           SolverStatus)
from pyomo.util.check_units import assert_units_consistent

from dispatches.case_studies.nuclear_case.nuclear_flowsheet \
    import build_ne_flowsheet, fix_dof_and_initialize
from idaes.core.solvers import get_solver


solver = get_solver()


@pytest.fixture(scope="module")
def build_npp():
    m = build_ne_flowsheet(np_capacity=1000, include_pem=False)
    fix_dof_and_initialize(m)

    return m


@pytest.fixture(scope="module")
def build_npp_pem():
    m = build_ne_flowsheet(np_capacity=1000, include_tank=False)
    fix_dof_and_initialize(m, split_frac_grid=0.8)

    return m


@pytest.fixture(scope="module")
def build_npp_pem_tank():
    m = build_ne_flowsheet(np_capacity=1000, include_turbine=False)
    fix_dof_and_initialize(m, split_frac_grid=0.8)

    return m


@pytest.fixture(scope="module")
def build_npp_pem_tank_turbine():
    m = ConcreteModel()

    # Build the nuclear flowsheet
    build_ne_flowsheet(m, np_capacity=1000)

    # Fix the degrees of freedom and initialize
    fix_dof_and_initialize(m, split_frac_grid=0.8, flow_mol_to_pipeline=10, flow_mol_to_turbine=10)

    return m


@pytest.fixture(scope="module")
def build_npp_pem_tank_turbine_capacity():
    m = ConcreteModel()

    # Build the nuclear flowsheet
    build_ne_flowsheet(
        m, 
        np_capacity=1000,
        pem_capacity=250,
        tank_capacity=4000,
        turbine_capacity=100,
    )

    # Fix the degrees of freedom and initialize
    fix_dof_and_initialize(m, split_frac_grid=0.8, flow_mol_to_pipeline=10, flow_mol_to_turbine=10)

    return m


@pytest.mark.unit
def test_npp(build_npp):
    m = build_npp

    assert not hasattr(m.fs, "pem")
    assert m.fs.np_power_split.np_to_pem_port.electricity[0].value == pytest.approx(0, rel=1e-2)


@pytest.mark.unit
def test_npp_pem(build_npp_pem):
    m = build_npp_pem

    assert hasattr(m.fs, "pem")
    assert not hasattr(m.fs, "h2_tank")

    # PEM System
    assert m.fs.pem.outlet.flow_mol[0].value == \
        pytest.approx(505.481, rel=1e-1)
    assert m.fs.pem.outlet.temperature[0].value == \
        pytest.approx(300, rel=1e-1)
    assert m.fs.pem.outlet.pressure[0].value == \
        pytest.approx(101325, rel=1e-1)


@pytest.mark.unit
def test_npp_pem_tank(build_npp_pem_tank):
    m = build_npp_pem_tank

    assert hasattr(m.fs, "pem")
    assert hasattr(m.fs, "h2_tank")
    assert not hasattr(m.fs, "translator")
    assert not hasattr(m.fs, "mixer")
    assert not hasattr(m.fs, "h2_turbine")

    # Hydrogen tank
    assert m.fs.h2_tank.outlet_to_turbine.flow_mol[0].value == pytest.approx(0, rel=1e-5)
    assert m.fs.h2_tank.tank_holdup_previous[0].value == \
        pytest.approx(0, rel = 1e-1)
    assert m.fs.h2_tank.tank_holdup[0].value == \
        pytest.approx(1747732.3199 + 36000, rel=1e-1)


@pytest.mark.unit
def test_npp_pem_tank_turbine(build_npp_pem_tank_turbine):
    m = build_npp_pem_tank_turbine

    results = solver.solve(m)

    # Ensure that units are consistent
    assert_units_consistent(m)

    # Check for optimal solution
    assert results.solver.termination_condition == \
        TerminationCondition.optimal
    assert results.solver.status == SolverStatus.ok

    # PEM System
    assert m.fs.pem.outlet.flow_mol[0].value == \
        pytest.approx(505.481, rel=1e-1)
    assert m.fs.pem.outlet.temperature[0].value == \
        pytest.approx(300, rel=1e-1)
    assert m.fs.pem.outlet.pressure[0].value == \
        pytest.approx(101325, rel=1e-1)
    
    # Hydrogen tank
    assert m.fs.h2_tank.tank_holdup_previous[0].value == \
        pytest.approx(0, rel = 1e-1)
    assert m.fs.h2_tank.tank_holdup[0].value == \
        pytest.approx(1747732.3199, rel=1e-1)

    # Hydrogen Turbine
    # Compressor
    assert m.fs.h2_turbine.compressor.outlet.temperature[0].value == \
        pytest.approx(793.42, rel=1e-1)

    # Stoichiometric Reactor
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'hydrogen'].value == \
        pytest.approx(0.00088043, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'nitrogen'].value == \
        pytest.approx(0.73278, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'oxygen'].value == \
        pytest.approx(0.15276, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'water'].value == \
        pytest.approx(0.1103, rel=1e-1)
    assert m.fs.h2_turbine.stoic_reactor.outlet.\
        mole_frac_comp[0, 'argon'].value == \
        pytest.approx(0.0032773, rel=1e-1)

    # Turbine
    assert m.fs.h2_turbine.turbine.inlet.temperature[0].value == \
        pytest.approx(1451.5, rel=1e-1)
    assert m.fs.h2_turbine.turbine.outlet.temperature[0].value == \
        pytest.approx(739.3, rel=1e-1)


@pytest.mark.unit
def test_npp_pem_tank_turbine_capacity(build_npp_pem_tank_turbine_capacity):
    m = build_npp_pem_tank_turbine_capacity

    results = solver.solve(m)

    assert m.fs.pem.electricity_in.electricity[0].ub == pytest.approx(250e3, rel=0.1)
    assert m.fs.h2_tank.tank_holdup_previous[0].ub == pytest.approx(4000 / 2.013e-3, rel=0.1)
    assert hasattr(m.fs.h2_turbine, "turbine_capacity")
    assert m.fs.h2_turbine.turbine_capacity.ub == pytest.approx(100e6, rel=0.1)
