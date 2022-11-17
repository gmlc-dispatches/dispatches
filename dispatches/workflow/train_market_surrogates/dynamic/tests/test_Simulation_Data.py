#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
# by the software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
# Research Corporation, et al.  All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and
# license information.
#################################################################################

import pytest
from pyomo.common import unittest as pyo_unittest
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
import idaes.logger as idaeslog
import os
import numpy as np
from pathlib import Path
try:
    # Pyton 3.8+
    from importlib import resources
except ImportError:
    # Python 3.7
    import importlib_resources as resources



def _get_data_path(file_name: str, package: str = "dispatches.workflow.train_market_surrogates.dynamic.tests.data") -> Path:
    with resources.path(package, file_name) as p:
        return Path(p)


@pytest.fixture
def sample_simulation_data() -> Path:
    return _get_data_path("simdatatest.csv")


@pytest.fixture
def sample_input_data():
    return _get_data_path("inputdatatest.h5")


@pytest.fixture
def num_sims():
    return 3


@pytest.fixture
def case_type():
    return 'NE'


@pytest.fixture
def fixed_pmax():
    return True


@pytest.fixture
def rev_data():
    return _get_data_path("revdatatest.csv")


@pytest.fixture
def base_simulationdata(sample_simulation_data, sample_input_data, num_sims, case_type, fixed_pmax):
    return SimulationData(sample_simulation_data, sample_input_data, num_sims, case_type, fixed_pmax)


@pytest.mark.unit
def test_create_SimulationData(sample_simulation_data, sample_input_data, num_sims, case_type, fixed_pmax):
    simulation_data = SimulationData(sample_simulation_data, sample_input_data, num_sims, case_type, fixed_pmax)
    assert simulation_data.dispatch_data_file is sample_simulation_data
    assert simulation_data.input_data_file is sample_input_data
    assert simulation_data.num_sims is num_sims
    assert simulation_data.case_type is case_type
    assert simulation_data.fixed_pmax is fixed_pmax


@pytest.mark.unit
def test_read_data_to_array(base_simulationdata):
    dispatch_array, index = base_simulationdata._read_data_to_array()
    expected_dispatch_array = np.array([np.ones(366*24)*200,np.ones(366*24)*340,np.ones(366*24)*400])
    expected_index = [0,1,2]

    # assertStructuredAlmostEqual do not support numpy.ndarray, use np.testing.assert_array_equal
    np.testing.assert_array_equal(dispatch_array,expected_dispatch_array)

    pyo_unittest.assertStructuredAlmostEqual(
        first=index, second=expected_index
    )


@pytest.mark.unit
def test_read_data_to_dict(base_simulationdata):
    test_dispatch_dict, test_input_dict = base_simulationdata.read_data_to_dict()
    expected_dispatch_dict = {0: np.ones(366*24)*200, 1: np.ones(366*24)*340, 2: np.ones(366*24)*400}
    expect_input_dict = {0: np.array([15,0.5,10,500]), 1: np.array([15,0.2,10,500]), 2: np.array([15,0.1,10,500])}

    # assertStructuredAlmostEqual do not support numpy.ndarray as values in dict, use np.testing.assert_array_equal
    for key_a,key_b in zip(test_dispatch_dict, expected_dispatch_dict):
        # compare if the keys are the same
        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        # compare if the values are the same
        np.testing.assert_array_equal(test_dispatch_dict[key_a], expected_dispatch_dict[key_b])

    for key_a, key_b in zip(test_input_dict, expect_input_dict):
        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(test_input_dict[key_a], expect_input_dict[key_b])



@pytest.mark.unit
def test_read_NE_pmin(base_simulationdata):
    test_pmin_dict = base_simulationdata._read_NE_pmin()
    expected_pmin_dict = {0: 200, 1: 320, 2: 360}

    pyo_unittest.assertStructuredAlmostEqual(
        first=test_pmin_dict, second=expected_pmin_dict
    )


@pytest.mark.unit
def test_scale_data(base_simulationdata):
    test_scaled_dispatch_dict = base_simulationdata._scale_data()
    expected_scaled_dispatch_dict = {0: np.zeros(366*24), 1: np.ones(366*24)*0.25, 2: np.ones(366*24)}

    for key_a, key_b in zip(test_scaled_dispatch_dict, expected_scaled_dispatch_dict):
        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(test_scaled_dispatch_dict[key_a], expected_scaled_dispatch_dict[key_b])


@pytest.mark.unit
def test_read_rev_data(base_simulationdata,rev_data):
    test_rev_dict = base_simulationdata.read_rev_data(rev_data)
    expected_rev_data = {0: 10000, 1: 20000, 2: 30000}

    pyo_unittest.assertStructuredAlmostEqual(
        first=test_rev_dict, second=expected_rev_data
    )
