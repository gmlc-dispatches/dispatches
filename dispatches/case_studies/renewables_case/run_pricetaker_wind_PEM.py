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
import multiprocessing as mp
from itertools import product
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dispatches.case_studies.renewables_case.wind_battery_PEM_LMP import wind_battery_pem_optimize
from dispatches.case_studies.renewables_case.RE_flowsheet import default_input_params, market, prices


# TempfileManager.tempdir = '/tmp/scratch'
file_dir = Path(__file__).parent / "wind_PEM"
if not file_dir.exists():
    os.mkdir(file_dir)

build_add_wind = 0 # if False, wind size is fixed. Either way, all wind capacity is part of capital cost

def run_design(h2_price, pem_ratio):
    input_params = default_input_params.copy()
    input_params['h2_price_per_kg'] = h2_price
    if pem_ratio == None:
        input_params['design_opt'] = False
    else:
        input_params['design_opt'] = "PEM"
    input_params["batt_mw"] = 0
    input_params["tank_size"] = 0
    input_params["pem_mw"] = pem_ratio * input_params["wind_mw"]
    if (file_dir / f"result_{market}_{h2_price}_{pem_ratio}.json").exists():
        with open(file_dir / f"result_{market}_{h2_price}_{pem_ratio}.json", 'r') as f:
            res = json.load(f)
        res = {**input_params, **res}
        res.pop("DA_LMPs")
        res.pop("design_opt")
        res.pop("extant_wind")
        res.pop("wind_resource")
        print(f"Already complete: {h2_price} {pem_ratio}")
        return res
    print(f"Running: {h2_price} {pem_ratio} {build_add_wind}")
    des_res = wind_battery_pem_optimize(
        time_points=24 * 7, 
        # time_points=len(prices), 
        input_params=input_params, verbose=False, plot=False)
    res = {**input_params, **des_res[0]}
    res.pop("DA_LMPs")
    res.pop("design_opt")
    res.pop("extant_wind")
    res.pop("wind_resource")
    res.pop("pyo_model")
    with open(file_dir / f"result_{market}_{h2_price}_{pem_ratio}.json", 'w') as f:
        json.dump(res, f)
    print(f"Finished: {h2_price} {pem_ratio}")
    return res

run_design(2.2, 0.2)
exit()

print(f"Writing to 'design_{market}_{build_add_wind}_results.csv'")
h2_prices = np.linspace(2, 3, 5)
pem_ratio = np.append(np.linspace(0, 1, 5), None)
# h2_prices = np.flip(h2_prices)
# price_cap = np.flip(price_cap)
inputs = product(h2_prices, pem_ratio)

with mp.Pool(processes=2) as p:
    res = p.starmap(run_design, inputs)

df = pd.DataFrame(res)
df.to_csv(file_dir / f"design_wind_PEM_results.csv")
