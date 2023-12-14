#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

# Pyton 3.8+
from importlib import resources
from pathlib import Path

import pytest

import numpy as np
from pyomo.common import unittest as pyo_unittest
import idaes.logger as idaeslog

pytest.importorskip("dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data")

from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData



def _get_data_path(file_name: str, package: str = "dispatches.workflow.train_market_surrogates.dynamic.tests.data") -> Path:
    with resources.path(package, file_name) as p:
        return Path(p)


@pytest.fixture
def sample_simulation_data() -> Path:
    return _get_data_path("simdatatest.csv")


@pytest.fixture
def sample_input_data_NE():
    return _get_data_path("input_data_test_NE.h5")


@pytest.fixture
def sample_input_data_RE():
    return _get_data_path("input_data_test_RE.h5")


@pytest.fixture
def sample_input_data_FE():
    return _get_data_path("input_data_test_FE.h5")


@pytest.fixture
def num_sims():
    return 3


# Create different case type for testing
@pytest.fixture
def case_type_NE():
    return 'NE'


@pytest.fixture
def case_type_RE():
    return 'RE'


@pytest.fixture
def case_type_FE():
    return 'FE'


@pytest.fixture
def rev_data():
    return _get_data_path("revdatatest.csv")


@pytest.fixture
def base_simulationdata_NE(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    return SimulationData(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE)


@pytest.fixture
def base_simulationdata_RE(sample_simulation_data, sample_input_data_RE, num_sims, case_type_RE):
    return SimulationData(sample_simulation_data, sample_input_data_RE, num_sims, case_type_RE)


@pytest.fixture
def base_simulationdata_FE(sample_simulation_data, sample_input_data_FE, num_sims, case_type_FE):
    return SimulationData(sample_simulation_data, sample_input_data_FE, num_sims, case_type_FE)


# some methods that are the consistent with 3 case studies are tested using NE.
@pytest.mark.unit
def test_create_SimulationData(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    simulation_data = SimulationData(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE)
    assert simulation_data.dispatch_data_file == sample_simulation_data
    assert simulation_data.input_data_file == sample_input_data_NE
    assert simulation_data.num_sims == num_sims
    assert simulation_data.case_type == case_type_NE


@pytest.mark.unit
def test_invalid_num_sims(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    sims = "10"
    with pytest.raises(TypeError, match=r".*The number of clustering years must be positive integer,*"):
        simulation_data = SimulationData(sample_simulation_data, sample_input_data_NE, sims, case_type_NE)


@pytest.mark.unit
def test_invalid_num_sims_2(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    sims = -1
    with pytest.raises(ValueError, match=r".*The number of simulation years must be positive integer,*"):
        simulation_data = SimulationData(sample_simulation_data, sample_input_data_NE, sims, case_type_NE)


@pytest.mark.unit
def test_valid_case_type(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    case_type = ["NE"]
    with pytest.raises(ValueError, match=r".*The value of case_type must be str*"):
        simulation_data = SimulationData(sample_simulation_data, sample_input_data_NE, num_sims, case_type)


@pytest.mark.unit
def test_valid_case_type(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    case_type = "BE"
    with pytest.raises(ValueError, match=r".*The case_type must be one of 'RE','NE' or 'FE',*"):
        simulation_data = SimulationData(sample_simulation_data, sample_input_data_NE, num_sims, case_type)


@pytest.mark.unit
def test_read_data_to_array(base_simulationdata_NE):
    dispatch_array, index = base_simulationdata_NE._read_data_to_array()
    expected_dispatch_array = np.array([np.ones(366*24)*200,np.ones(366*24)*340,np.ones(366*24)*400])
    expected_index = [0,1,2]

    # assertStructuredAlmostEqual do not support numpy.ndarray, use np.testing.assert_array_equal
    np.testing.assert_array_equal(dispatch_array,expected_dispatch_array)

    pyo_unittest.assertStructuredAlmostEqual(
        first=index, second=expected_index
    )


@pytest.mark.unit
def test_read_data_to_dict(base_simulationdata_NE):
    test_dispatch_dict, test_input_dict = base_simulationdata_NE.read_data_to_dict()
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
def test_read_NE_pmin(base_simulationdata_NE):
    test_pmin_dict = base_simulationdata_NE._read_NE_pmin()
    expected_pmin_dict = {0: 200, 1: 320, 2: 360}

    pyo_unittest.assertStructuredAlmostEqual(
        first=test_pmin_dict, second=expected_pmin_dict
    )


@pytest.mark.unit
def test_read_RE_pmax(base_simulationdata_RE):
    test_pmax = base_simulationdata_RE._read_RE_pmax(wind_gen = '303_WIND_1')
    expected_pmax = 847.0
    pyo_unittest.assertStructuredAlmostEqual(
        first=test_pmax, second=expected_pmax
    )


@pytest.mark.unit
def test_invalid_RE_gen_name(base_simulationdata_RE):
    with pytest.raises(NameError, match=r".*wind generator name*"):
        test_pmax = base_simulationdata_RE._read_RE_pmax(wind_gen = '111_WIND_1')


@pytest.mark.unit
def test_read_FE_pmax(base_simulationdata_FE):
    test_pmax = base_simulationdata_FE._read_FE_pmax()
    expected_pmax_dict = {0:451, 1:451, 2:466}
    pyo_unittest.assertStructuredAlmostEqual(
        first=test_pmax, second=expected_pmax_dict
    )


@pytest.mark.unit
def test_scale_data_NE(base_simulationdata_NE):
    test_scaled_dispatch_dict = base_simulationdata_NE._scale_data()
    expected_scaled_dispatch_dict = {0: np.zeros(366*24), 1: np.ones(366*24)*0.25, 2: np.ones(366*24)}

    for key_a, key_b in zip(test_scaled_dispatch_dict, expected_scaled_dispatch_dict):
        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(test_scaled_dispatch_dict[key_a], expected_scaled_dispatch_dict[key_b])


@pytest.mark.unit
def test_scale_data_RE(base_simulationdata_RE):
    test_scaled_dispatch_dict = base_simulationdata_RE._scale_data()
    expected_scaled_dispatch_dict = {0: np.ones(366*24)*200/847, 1: np.ones(366*24)*340/847, 2: np.ones(366*24)*400/847}

    for key_a, key_b in zip(test_scaled_dispatch_dict, expected_scaled_dispatch_dict):
        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(test_scaled_dispatch_dict[key_a], expected_scaled_dispatch_dict[key_b])


@pytest.mark.unit
def test_scale_data_FE(base_simulationdata_FE):
    test_scaled_dispatch_dict = base_simulationdata_FE._scale_data()
    expected_scaled_dispatch_dict = {0: np.ones(366*24)*-84/152, 1: np.ones(366*24)*56/152, 2: np.ones(366*24)*116/152}

    for key_a, key_b in zip(test_scaled_dispatch_dict, expected_scaled_dispatch_dict):
        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(test_scaled_dispatch_dict[key_a], expected_scaled_dispatch_dict[key_b])


@pytest.mark.unit
def test_read_rev_data(base_simulationdata_NE,rev_data):
    test_rev_dict = base_simulationdata_NE.read_rev_data(rev_data)
    expected_rev_data = {0: 10000, 1: 20000, 2: 30000}

    pyo_unittest.assertStructuredAlmostEqual(
        first=test_rev_dict, second=expected_rev_data
    )


@pytest.mark.unit
def test_read_wind_data(base_simulationdata_RE):
    wind_data = base_simulationdata_RE.read_wind_data()
    data_shape = np.shape(wind_data)
    expect_shape = (366, 24)

    pyo_unittest.assertStructuredAlmostEqual(
        first=data_shape, second=expect_shape
    )

