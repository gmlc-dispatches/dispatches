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

import pytest

from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
)
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
from dispatches.case_studies.fossil_case.supercritical_plant.supercritical_powerplant import (
    build_scpc_flowsheet,
    fix_dof_and_initialize,
    unfix_dof_for_optimization,
)


@pytest.mark.unit
def test_scpc_without_tes():
    m = build_scpc_flowsheet(include_concrete_tes=False)

    # Check all units are present
    assert not hasattr(m.fs, "tes")
    assert not hasattr(m.fs, "discharge_turbine")
    assert hasattr(m.fs, "boiler")
    assert hasattr(m.fs, "reheater")
    assert hasattr(m.fs, "hp_splitter")
    assert hasattr(m.fs, "bfpt")
    assert hasattr(m.fs, "turbine") and len(m.fs.turbine) == 9
    assert hasattr(m.fs, "t_splitter") and len(m.fs.t_splitter) == 8

    fix_dof_and_initialize(m, outlvl=idaeslog.WARNING)

    res = get_solver().solve(m)
    assert_optimal_termination(res)

    unfix_dof_for_optimization(m)

    assert (m.fs.net_power_output[0].value / 1e6) == pytest.approx(692, abs=1)


@pytest.mark.unit
def test_scpc_with_tes():
    m = ConcreteModel()
    m = build_scpc_flowsheet(m=m, include_concrete_tes=True)

    # Check all units are present
    assert hasattr(m.fs, "tes")
    assert hasattr(m.fs, "discharge_turbine")

    fix_dof_and_initialize(m)

    res = get_solver().solve(m)
    assert_optimal_termination(res)

    unfix_dof_for_optimization(m)

    assert (m.fs.net_power_output[0].value / 1e6) == pytest.approx(625, abs=1)
