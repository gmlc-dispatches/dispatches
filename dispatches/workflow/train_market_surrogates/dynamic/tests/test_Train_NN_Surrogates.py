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
pytest.importorskip("dispatches.workflow.train_market_surrogates.dynamic.Train_NN_Surrogates")

from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.Train_NN_Surrogates import TrainNNSurrogates


def _get_data_path(file_name: str, package: str = "dispatches.workflow.train_market_surrogates.dynamic.tests.data") -> Path:
    with resources.path(package, file_name) as p:
        return Path(p)

@pytest.fixture
def sample_simulation_data() -> Path:
    return _get_data_path("simdatatest.csv")


@pytest.fixture
def sample_input_data():
    return _get_data_path("input_data_test_NE.h5")


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
def filter_opt():
    return True


@pytest.fixture
def data_file():
    return _get_data_path("sample_clustering_model.json")


@pytest.fixture
def base_simulationdata(sample_simulation_data, sample_input_data, num_sims, case_type):
    return SimulationData(sample_simulation_data, sample_input_data, num_sims, case_type)


@pytest.fixture
def base_NNtrainer(base_simulationdata, data_file, filter_opt):
    return TrainNNSurrogates(base_simulationdata, str(data_file), filter_opt)


@pytest.mark.unit
def test_invalid_simulation_data(base_simulationdata, data_file, filter_opt):
    invalid_simulation_data = "simulation_data"
    with pytest.raises(TypeError, match=r".*The simulation_data must be created from SimulationData.*"):
        tnn = TrainNNSurrogates(invalid_simulation_data, data_file, filter_opt)


@pytest.mark.unit
def test_invalid_data_file(base_simulationdata, data_file, filter_opt):
    invalid_data_file = 123
    with pytest.raises(TypeError, match=r".*The data_file must be str or object, but*"):
        tnn = TrainNNSurrogates(base_simulationdata, invalid_data_file, filter_opt)


@pytest.mark.unit
def test_invalid_filter_opt(base_simulationdata, data_file, filter_opt):
    invalid_filter_opt = "True"
    with pytest.raises(TypeError, match=r".*Filter_opt must be bool, but*"):
        tsc = TrainNNSurrogates(base_simulationdata, data_file, invalid_filter_opt)


@pytest.mark.unit
def test_create_TrainNNSurrogates(base_simulationdata, data_file, filter_opt):
    NNtrainer = TrainNNSurrogates(base_simulationdata, str(data_file), filter_opt)
    assert NNtrainer.simulation_data == base_simulationdata
    assert NNtrainer.data_file == str(data_file)
    assert NNtrainer.filter_opt == filter_opt
    

@pytest.mark.unit
def test_read_clustering_model(base_NNtrainer, data_file):
    base_NNtrainer._read_clustering_model(str(data_file))
    num_cluster = base_NNtrainer.num_clusters
    expected_num_cluster = 1

    pyo_unittest.assertStructuredAlmostEqual(
        first=num_cluster, second=expected_num_cluster
    )


@pytest.mark.unit
def test_generate_label_data(base_NNtrainer,data_file):
    # in _read_clustering_model() the funciton will set the self.num_clusters
    base_NNtrainer._read_clustering_model(str(data_file))
    dispatch_frequency_dict = base_NNtrainer._generate_label_data()
    expected_dispatch_frequency_dict = {0:[1,0,0], 1:[0,1,0], 2:[0,0,1]}
    
    pyo_unittest.assertStructuredAlmostEqual(
        first=dispatch_frequency_dict, second=expected_dispatch_frequency_dict
    )

