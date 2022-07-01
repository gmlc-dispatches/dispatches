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
import numpy as np
import copy
from pathlib import Path
from PySAM.ResourceTools import SRW_to_wind_data
from functools import partial


timestep_hrs = 1                # timestep [hr]
# constants
h2_mols_per_kg = 500
H2_mass = 2.016 / 1000

# costs in per kW unless specified otherwise
wind_cap_cost = 1550
wind_op_cost = 43
batt_cap_cost = 300 * 4                     # per kW for 4 hour battery
pem_cap_cost = 1630
pem_op_cost = 47.9
pem_var_cost = 1.3/1000                     # per kWh
tank_cap_cost_per_m3 = 29 * 0.8 * 1000      # per m^3
tank_cap_cost_per_kg = 29 * 33.5            # per kg
tank_op_cost = .17 * tank_cap_cost_per_kg   # per kg
turbine_cap_cost = 1000
turbine_op_cost = 11.65
turbine_var_cost = 4.27/1000                # per kWh

# prices
h2_price_per_kg = 2

# sizes
fixed_wind_mw = 847
wind_mw_ub = 10000
fixed_batt_mw = 5761
fixed_pem_mw = 643
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

# prices
with open(Path(__file__).parent / 'rts_results_all_prices.npy', 'rb') as f:
    dispatch = np.load(f)
    price = np.load(f)

prices_used = copy.copy(price)
prices_used[prices_used > 200] = 200
weekly_prices = prices_used.reshape(52, 168)
# n_time_points = 7 * 24

# simple financial assumptions
i = 0.05                                    # discount rate
N = 30                                      # years
PA = ((1+i)**N - 1)/(i*(1+i)**N)            # present value / annuity = 1 / CRF

# wind resource data from example Wind Toolkit file
wind_data = SRW_to_wind_data(Path(__file__).parent / '44.21_-101.94_windtoolkit_2012_60min_80m.srw')
wind_speeds = [wind_data['data'][i][2] for i in range(8760)]

wind_resource = {t:
                    {'wind_resource_config': {
                         'resource_speed': [wind_speeds[t]]
                    }
                } for t in range(8760)}

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

    "wind_resource": wind_resource,
    "h2_price_per_kg": h2_price_per_kg,
    "DA_LMPs": prices_used,

    "design_opt": True,
    "extant_wind": True
}