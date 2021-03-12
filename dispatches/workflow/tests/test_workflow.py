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
    df.create(hello="ignored")
    # with unknown type, raises KeyError
    pytest.raises(KeyError, DatasetFactory, "?")