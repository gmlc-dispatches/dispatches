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
from argparse import ArgumentParser
import pyomo.environ as pyo
from itertools import product
import os
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dispatches.case_studies.renewables_case.wind_battery_LMP import wind_battery_optimize
from dispatches.case_studies.renewables_case.RE_flowsheet import default_input_params, market

usage = "Run pricetaker optimization with battery size and duration."
parser = ArgumentParser(usage)

parser.add_argument(
    "--battery_ratio",
    dest="battery_ratio",
    help="Indicate the battery ratio to the wind farm.",
    action="store",
    type=float,
    default=0.1,
)

parser.add_argument(
    "--duration",
    dest="duration",
    help="the battery duration hours",
    action="store",
    type=int,
    default=4,
)

parser.add_argument(
    "--year",
    dest="year",
    help="The year of the capital and OM cost price we are using.",
    action="store",
    type=int,
    default=2023,
)

options = parser.parse_args()
battery_ratio = options.battery_ratio
duration = options.duration
year = options.year

lmps_df = pd.read_parquet(Path(__file__).parent / "data" / "303_LMPs_15_reserve_500_shortfall.parquet")
lmps = lmps_df['LMP DA'].values
lmps[lmps>500] = 500
default_input_params['DA_LMPs'] = lmps # even we use rt lmp signals, we call it DA_LMPs to simplify the work.

# TempfileManager.tempdir = '/tmp/scratch'
file_dir = Path(__file__).parent / f"wind_battery_pricetaker_rerun_{year}"
if not file_dir.exists():
    os.mkdir(file_dir)

build_add_wind = True # if False, wind size is fixed. Either way, all wind capacity is part of capital cost

def run_design(wind_size, battery_ratio, duration = 4):
    input_params = default_input_params.copy()
    input_params["design_opt"] = False
    input_params["extant_wind"] = True
    input_params["wind_mw"] = wind_size
    input_params["batt_mw"] = battery_ratio*wind_size
    input_params["tank_size"] = 0
    if (file_dir / f"result_{market}_wind_{wind_size}_battery_{battery_ratio}_hour_{duration}.json").exists():
        with open(file_dir / f"result_{market}_wind_{wind_size}_battery_{battery_ratio}_hour_{duration}.json", 'r') as f:
            res = json.load(f)
        res = {**input_params, **res}
        res.pop("DA_LMPs")
        res.pop("design_opt")
        res.pop("extant_wind")
        res.pop("wind_resource")
        print(f"Already complete: {wind_size} {battery_ratio}")
    
        return res

    print(f"Running: {wind_size} {battery_ratio} {build_add_wind}")
    des_res = wind_battery_optimize(
        n_time_points=8736, 
        # n_time_points=len(lmps_df), 
        input_params=input_params, verbose=True)
    # res = {**input_params, **des_res[0]}
    res = {**input_params,"NPV": pyo.value(des_res.NPV), "annual elec revenue":pyo.value(des_res.annual_elec_revenue), "annual revenue": pyo.value(des_res.annual_revenue), "total elec output": pyo.value(des_res.total_elec_output)}
    res.pop("DA_LMPs")
    res.pop("design_opt")
    res.pop("extant_wind")
    res.pop("wind_resource")
    res.pop("pyo_model")
    with open(file_dir / f"result_{market}_wind_{wind_size}_battery_{battery_ratio}_hour_{duration}.json", 'w') as f:
        json.dump(res, f)
    print(f"Finished: {wind_size} {battery_ratio}")
    
    return res
