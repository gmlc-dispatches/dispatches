from pathlib import Path
import pickle

import pytest

keras = pytest.importorskip("keras", reason=f"keras required to run {__file__}")

from dispatches.case_studies.renewables_case.RE_surrogate_optimization_steadystate import re_nn_dir


@pytest.fixture(scope="module")
def base_dir() -> Path:
    return re_nn_dir


def test_base_dir(base_dir: Path):
    assert base_dir.is_dir()
    contents = sorted(base_dir.rglob("*"))
    assert len(contents) > 0


class TestModelDeserialization:

    def test_revenue(self, base_dir: Path):
        model = keras.models.load_model(base_dir / "revenue" / "RE_revenue_2_25")

        from keras.engine.sequential import Sequential
        assert isinstance(model, Sequential)

    def test_clustering(self, base_dir: Path):
        with (base_dir / "dispatch_frequency" / "static_clustering_wind_pmax.pkl").open("rb") as f:
            model = pickle.load(f)

        from sklearn.cluster import KMeans
        assert isinstance(model, KMeans)