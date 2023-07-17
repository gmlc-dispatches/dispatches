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
import numpy as np
import copy
from pathlib import Path
import pandas as pd
from PySAM.ResourceTools import SRW_to_wind_data
from pyomo.common.fileutils import this_file_dir
import json

re_case_dir = Path(this_file_dir())

timestep_hrs = 1                            # timestep [hr]
# constants
h2_mols_per_kg = 500
H2_mass = 2.016 / 1000
kg_to_tons = 0.00110231

with open(re_case_dir/"wind_battery_cost_parameter.json", "rb") as f:
    price_dict = json.load(f)

# only need to change the year and scenario here, default, 2023, moderate, 4hr battery.
year = 2023
scenario = "moderate"
duration = 4
duration_list = [2, 4, 6, 8, 10] 
arg_duration = int(duration/2 - 1)

batt_op_cost = price_dict["battery"]["fixed_om"][scenario][str(year)][arg_duration]     # per kw-yr for duration-hr battery
batt_cap_cost_kw = price_dict["battery"]["batt_cap_cost_param"][scenario][str(year)][0]    # per kW for duration-hr battery
batt_cap_cost_kwh = price_dict["battery"]["batt_cap_cost_param"][scenario][str(year)][1]    # per kW for duration-hr battery

wind_cap_cost = price_dict["wind"]["capital"][scenario][str(year)][0]    # per kW
wind_op_cost = price_dict["wind"]["fixed_om"][scenario][str(year)][0]    # per kW-hr

# costs in per kW unless specified otherwise
batt_rep_cost_kwh = batt_cap_cost_kw * 0.5 / 4 # assume 50% price w/ discounting and 4 hour battery
pem_cap_cost = 1200
pem_op_cost = 0.03 * pem_cap_cost
pem_var_cost = 0                            # per kWh
tank_cap_cost_per_m3 = 29 * 0.8 * 1000      # per m^3
tank_cap_cost_per_kg = 29 * 33.5            # per kg
tank_op_cost = .17 * tank_cap_cost_per_kg   # per kg
turbine_cap_cost = 1320
turbine_op_cost = 11.65
turbine_var_cost = 4.27/1000                # per kWh

# prices
h2_price_per_kg = 2

# sizes
fixed_wind_mw = 847
wind_mw_ub = 10000
fixed_batt_mw = 100
fixed_pem_mw = 400
turb_p_mw = 1
valve_cv = 0.00001
fixed_tank_size = 0.5

# operation parameters
pem_bar = 1.01325
pem_temp = 300                  # [K]
# battery_ramp_rate = 25 * 1e3              # kwh/hr
battery_ramp_rate = 1e8
h2_turb_min_flow = 1e-3
air_h2_ratio = 10.76
compressor_dp = 24.01
max_pressure_bar = 700

# load pre-compiled RTS-GMLC output data
df = pd.read_csv(re_case_dir / "data" / "Wind_Thermal_Dispatch.csv")
df.index = pd.to_datetime(df["DateTime"])

# drop indices not in original data set
start_date = pd.Timestamp('2020-01-02 00:00:00')
ix = pd.date_range(start=start_date, 
                    end=start_date
                    + pd.offsets.DateOffset(days=365)
                    - pd.offsets.DateOffset(hours=1),
                    freq='1H')
ix = ix[(ix.day != 29) | (ix.month != 2)]

df = df[df.index.isin(ix)]

bus = "303"
market = "DA"
if market == "Both":
    prices = np.max((df[f"{bus}_DALMP"].values, df[f"{bus}_RTLMP"].values), axis=0)
else:
    prices = df[f"{bus}_{market}LMP"].values
prices_used = copy.copy(prices)
# prices_used[prices_used > 200] = 200
weekly_prices = prices_used.reshape(52, 168)
# n_time_points = 7 * 24

n_timesteps = len(prices)

if market == "Both":
    wind_cfs = df[f"{bus}_WIND_1-RTCF"].values
else:
    wind_cfs = df[f"{bus}_WIND_1-{market}CF"].values

wind_capacity_factors = {t:
                            {'wind_resource_config': {
                                'capacity_factor': 
                                    [wind_cfs[t]]}} for t in range(n_timesteps)}
# simple financial assumptions
discount_rate = 0.08                                    # discount rate
N = 30                                                  # years
PA = ((1+discount_rate)**N - 1)/(discount_rate*(1+discount_rate)**N)            # present value / annuity = 1 / CRF

default_input_params = {
    "wind_mw": fixed_wind_mw,
    "wind_mw_ub": wind_mw_ub,
    "batt_mw": fixed_batt_mw,
    "pem_mw": fixed_pem_mw,
    "pem_bar": pem_bar,
    "pem_temp": pem_temp,
    "tank_size": fixed_tank_size,
    "tank_type": "simple",
    "turb_mw": turb_p_mw,

    "wind_resource": wind_capacity_factors,
    "h2_price_per_kg": h2_price_per_kg,
    "DA_LMPs": prices_used,

    "design_opt": True,
    "extant_wind": True
}
