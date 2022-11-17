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
from dispatches.workflow.train_market_surrogates.dynamic.Train_NN_Surrogates import TrainNNSurrogates
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
def sample_simulation_data_path() -> Path:
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
def filter_opt():
    return True


@pytest.fixture
def data_file():
    return _get_data_path("sample_clustering_model.json")


@pytest.fixture
def base_simulationdata(sample_simultaion_data, sample_input_data, num_sims, case_type, fixed_pmax):
    return SimulationData(sample_simultaion_data, sample_input_data, num_sims, case_type, fixed_pmax)


@pytest.fixture
def base_NNtrainer(base_simulationdata, data_file, filter_opt):
    return TrainNNSurrogates(base_simulationdata, data_file, filter_opt)


@pytest.mark.unit
def test_create_TrainNNSurrogates(base_simulationdata, data_file, filter_opt):
    NNtrainer = TrainNNSurrogates(base_simulationdata, data_file, filter_opt)
    assert NNtrainer.simulation_data is base_simulationdata
    assert NNtrainer.data_file is data_file
    assert NNtrainer.filter_opt is filter_opt
    

@pytest.mark.unit
def test_read_clustering_model(base_NNtrainer, data_file):
    base_NNtrainer._read_clustering_model(data_file)
    num_cluster = base_NNtrainer.num_clusters
    expected_num_cluster = 1

    pyo_unittest.assertStructuredAlmostEqual(
        first=num_cluster, second=expected_num_cluster
    )


@pytest.mark.unit
def test_generate_label_data(base_NNtrainer,data_file):
    # in _read_clustering_model() the funciton will set the self.num_clusters
    base_NNtrainer._read_clustering_model(data_file)
    dispatch_frequency_dict = base_NNtrainer._generate_label_data()
    expected_dispatch_frequency_dict = {0:[1,0,0], 1:[0,1,0], 2:[0,0,1]}
    
    pyo_unittest.assertStructuredAlmostEqual(
        first=dispatch_frequency_dict, second=expected_dispatch_frequency_dict
    )

