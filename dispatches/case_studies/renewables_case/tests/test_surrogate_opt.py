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
import pytest
pytest.importorskip("tensorflow", reason="optional dependencies for surrogate modeling not available")

from dispatches.case_studies.renewables_case.RE_surrogate_optimization_steadystate import *

def load_model():
    net_rev_defn, net_frequency_defn, dispatch_clusters_mean, pem_clusters_mean, resource_clusters_mean = load_surrogate_model(re_nn_dir)
    assert len([l for l in net_rev_defn.layers]) == 4
    assert len([l for l in net_frequency_defn.layers]) == 5
    assert dispatch_clusters_mean.mean() == pytest.approx(0.2445, rel=1e-3)
    assert pem_clusters_mean.mean() == pytest.approx(0.2215, rel=1e-3)
    assert resource_clusters_mean.mean() == pytest.approx(0.5547, rel=1e-3)

def test_RE_surrogate_steady_state_fixed():
    results = run_design(PEM_bid=30, PEM_size=200)
    assert results['e_revenue'] == pytest.approx(17901755, rel=1e-3)
    assert results['h_revenue'] == pytest.approx(47670633, rel=1e-3)
    assert results['NPV'] == pytest.approx(18766297, rel=1e-3)

def test_RE_surrogate_steady_state():
    results = run_design()
    assert results['pem_mw'] == pytest.approx(249.34, rel=1e-3)
    assert results['pem_bid'] == pytest.approx(26.21, rel=1e-3)
    assert results['e_revenue'] == pytest.approx(18137487, rel=1e-3)
    assert results['h_revenue'] == pytest.approx(55074013, rel=1e-3)
    assert results['NPV'] == pytest.approx(25546243, rel=1e-3)
