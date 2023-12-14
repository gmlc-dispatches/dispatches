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
import os
from pathlib import Path

import pytest
from pyomo.common import unittest as pyo_unittest
import idaes.logger as idaeslog
import numpy as np

pytest.importorskip("dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data")
pytest.importorskip("dispatches.workflow.train_market_surrogates.dynamic.Time_Series_Clustering")

from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.Time_Series_Clustering import TimeSeriesClustering


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
def fixed_pmax():
    return True


@pytest.fixture
def num_clusters():
    return 1


@pytest.fixture
def filter_opt():
    return True


@pytest.fixture
def filter_opt_F():
    return False


@pytest.fixture
def metric():
    return 'euclidean'


@pytest.fixture
def sample_clustering_results():
    return _get_data_path("sample_clustering_model.json")


@pytest.fixture
def base_simulationdata_NE(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE):
    return SimulationData(sample_simulation_data, sample_input_data_NE, num_sims, case_type_NE)


@pytest.fixture
def base_timeseriesclustering_NE(base_simulationdata_NE, num_clusters, filter_opt, metric):
    return TimeSeriesClustering(base_simulationdata_NE, num_clusters, filter_opt, metric)

@pytest.fixture
def base_simulationdata_RE(sample_simulation_data, sample_input_data_RE, num_sims, case_type_RE):
    return SimulationData(sample_simulation_data, sample_input_data_RE, num_sims, case_type_RE)


@pytest.fixture
def base_timeseriesclustering_RE(base_simulationdata_RE, num_clusters, filter_opt_F, metric):
    return TimeSeriesClustering(base_simulationdata_RE, num_clusters, filter_opt_F, metric)


@pytest.fixture
def base_simulationdata_FE(sample_simulation_data, sample_input_data_FE, num_sims, case_type_FE):
    return SimulationData(sample_simulation_data, sample_input_data_FE, num_sims, case_type_FE)


@pytest.fixture
def base_timeseriesclustering_FE(base_simulationdata_FE, num_clusters, filter_opt, metric):
    return TimeSeriesClustering(base_simulationdata_FE, num_clusters, filter_opt, metric)


@pytest.mark.unit
def test_create_TimeSeriesClustering(base_simulationdata_NE, num_clusters, filter_opt, metric):
    tsc = TimeSeriesClustering(base_simulationdata_NE, num_clusters, filter_opt, metric)
    assert tsc.num_clusters == num_clusters
    assert tsc.simulation_data == base_simulationdata_NE
    assert tsc.filter_opt == filter_opt
    assert tsc.metric == metric


@pytest.mark.unit
def test_create_RE_with_filter(base_simulationdata_RE, num_clusters, filter_opt, metric):
    with pytest.raises(TypeError, match=r".*cannot have set the filter_opt to*"):
        tsc = TimeSeriesClustering(base_simulationdata_RE, num_clusters, filter_opt, metric)


@pytest.mark.unit
def test_invalid_simulation_data(base_simulationdata_NE, num_clusters, filter_opt, metric):
    invalid_simulation_data = "simulation_data"
    with pytest.raises(TypeError, match=r".*The simulation_data must be created from SimulationData.*"):
        tsc = TimeSeriesClustering(invalid_simulation_data, num_clusters, filter_opt, metric)


@pytest.mark.unit
def test_invalid_metric(base_simulationdata_NE, num_clusters, filter_opt, metric):
    invalid_metric = "abc"
    with pytest.raises(ValueError, match=r".*The metric must be one of euclidean or dtw, but*"):
        tsc = TimeSeriesClustering(base_simulationdata_NE, num_clusters, filter_opt, invalid_metric)


@pytest.mark.unit
def test_invalid_num_clusters(base_simulationdata_NE, num_clusters, filter_opt, metric):
    invalid_num_clusters = "123"
    with pytest.raises(TypeError, match=r".*Number of clusters must be integer, but*"):
        tsc = TimeSeriesClustering(base_simulationdata_NE, invalid_num_clusters, filter_opt, metric)


@pytest.mark.unit
def test_invalid_filter_opt(base_simulationdata_NE, num_clusters, filter_opt, metric):
    invalid_filter_opt = "True"
    with pytest.raises(TypeError, match=r".*Filter_opt must be bool, but*"):
        tsc = TimeSeriesClustering(base_simulationdata_NE, num_clusters, invalid_filter_opt, metric)


@pytest.mark.unit
def test_transform_data_NE(base_timeseriesclustering_NE):
    train_data = base_timeseriesclustering_NE._transform_data()
    # test on the shape of the data to see if the filter is working. 
    data_shape = np.shape(train_data)
    expect_data_shape = (366,24,1)

    pyo_unittest.assertStructuredAlmostEqual(
        first=data_shape, second=expect_data_shape
    )


@pytest.mark.unit
def test_transform_data_RE(base_timeseriesclustering_RE):
    train_data = base_timeseriesclustering_RE._transform_data()
    # test on the shape of the data to see if the filter is working. 
    data_shape = np.shape(train_data)
    expect_data_shape = (366*3,2,24)

    pyo_unittest.assertStructuredAlmostEqual(
        first=data_shape, second=expect_data_shape
    )


@pytest.mark.unit
def test_transform_data_FE(base_timeseriesclustering_FE):
    train_data = base_timeseriesclustering_FE._transform_data()
    # test on the shape of the data to see if the filter is working. 
    data_shape = np.shape(train_data)
    expect_data_shape = (366*3,24,1)

    pyo_unittest.assertStructuredAlmostEqual(
        first=data_shape, second=expect_data_shape
    )


@pytest.mark.unit
def test_get_cluster_centers(base_timeseriesclustering_NE, sample_clustering_results):
    centers_dict = base_timeseriesclustering_NE.get_cluster_centers(sample_clustering_results)
    a = []
    for i in range(24):
        a.append([0.25])
    
    expected_centers_dict = {0:np.array(a)}

    for key_a, key_b in zip(centers_dict,expected_centers_dict):

        pyo_unittest.assertStructuredAlmostEqual(
            first=key_a, second=key_b
        )
        np.testing.assert_array_equal(centers_dict[key_a], expected_centers_dict[key_b])
