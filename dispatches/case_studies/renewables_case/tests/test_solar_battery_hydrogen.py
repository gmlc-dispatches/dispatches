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

from dispatches.case_studies.renewables_case.solar_battery_hydrogen_inputs import re_h2_parameters
from dispatches.case_studies.renewables_case.solar_battery_hydrogen import pv_battery_hydrogen_optimize


def test_solar_batt_hydrogen_fixed_design():
    params = re_h2_parameters.copy()
    params['max_sales'] = 1000
    params['max_purchases'] = 1000
    params['design_opt'] = False
    des_res, df_res = pv_battery_hydrogen_optimize(
        n_time_points=len(params['pv_resource']),
        input_params=params, verbose=False, plot=False)
    assert des_res['tank_tonH2'] == pytest.approx(8, rel=0.1)
    assert des_res['capital_cost'] == pytest.approx(3839511, abs=1)
    assert des_res['NPV'] == pytest.approx(-128628419, rel=1e-2)

def test_solar_batt_hydrogen_optimize():
    params = re_h2_parameters.copy()
    params['pv_mw'] = 0
    params['turb_mw'] = 0
    params['max_sales'] = 1000
    params['max_purchases'] = 1000
    params['design_opt'] = True
    des_res, df_res = pv_battery_hydrogen_optimize(
        n_time_points=len(params['pv_resource']),
        input_params=params, verbose=False, plot=False)
    assert des_res['pv_mw'] == pytest.approx(4, rel=0.1)
    assert des_res['batt_mw'] == pytest.approx(303, rel=0.1)
    assert des_res['batt_mwh'] == pytest.approx(151, rel=0.1)
    assert des_res['pem_mw'] == pytest.approx(0, abs=1)
    assert des_res['tank_tonH2'] == pytest.approx(0, abs=1)
    assert des_res['capital_cost'] == pytest.approx(60383533, rel=1e-2)
    assert des_res['NPV'] == pytest.approx(-214464199, rel=1e-2)
