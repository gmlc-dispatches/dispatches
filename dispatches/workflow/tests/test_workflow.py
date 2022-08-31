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
Tests for dispatches.workflow.workflow
"""
import pytest

from dispatches.workflow import ManagedWorkflow
from dispatches.workflow.workflow import Dataset, DatasetFactory


def test_managed_workflow():
    wf = ManagedWorkflow("hello", "world")
    assert wf.name == "hello"
    assert wf.workspace_name == "world"


def test_dataset():
    ds = Dataset("hello")
    assert ds.name == "hello"
    assert ds.meta == {}


def test_dataset_factory():
    df = DatasetFactory("null")
    assert df.create(hello="ignored") is None
    # with unknown type, raises KeyError
    pytest.raises(KeyError, DatasetFactory, "?")
