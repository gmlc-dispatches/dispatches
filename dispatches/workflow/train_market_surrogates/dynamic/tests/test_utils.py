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
def sample_raw_dispatch_data() -> Path:
    return _get_data_path("sample_raw_data")