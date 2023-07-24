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

    @pytest.mark.parametrize(
        "name",
        [
            "RE_revenue_2_25",
        ]
    )
    def test_revenue(self, base_dir: Path, name: str):
        model = keras.models.load_model(base_dir / "revenue" / name)

        from keras.engine.sequential import Sequential
        assert isinstance(model, Sequential)

    @pytest.mark.parametrize(
        "name",
        [
            "ss_surrogate_model_wind_pmax",
        ]
    )
    def test_dispatch(self, base_dir: Path, name: str):
        model = keras.models.load_model(base_dir / "dispatch_frequency" / name)

        from keras.engine.sequential import Sequential
        assert isinstance(model, Sequential)

    def test_clustering(self, base_dir: Path):
        with (base_dir / "dispatch_frequency" / "static_clustering_wind_pmax.pkl").open("rb") as f:
            model = pickle.load(f)

        from sklearn.cluster import KMeans
        assert isinstance(model, KMeans)