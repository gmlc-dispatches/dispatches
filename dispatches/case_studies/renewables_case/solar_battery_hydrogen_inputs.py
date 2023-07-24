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
import pandas as pd
from pyomo.common.fileutils import this_file_dir

from dispatches.case_studies.renewables_case.load_parameters import *

# settings
market = "DA"
h2_blend_ratio = 1.0                        # h2 to ng

data_dir = Path(this_file_dir()) / ".."

tax_incentives = 0.50

pv_cap_cost = 1420 * tax_incentives         # kwAC
pv_op_cost = 21                             # kwAC-yr
batt_cap_cost_kw = 236.36 * tax_incentives  # Fitted from NREL ATB 2021
batt_cap_cost_kwh = 254.83 * tax_incentives   
pem_cap_cost_kw = 1240
pem_op_cost = 47.9
pem_var_cost = 1.3/1000                     # per kWh
turbine_cap_cost = 1320                     # Hydrogen-fired gas turbine
turbine_op_cost = 11.65
turbine_var_cost = 3/1000
tank_cap_cost_per_kg = 500                  # On-site Bulk Storage with Current Technology 700 bar compressed (Type IV, single tank)
tank_op_cost = .17 * tank_cap_cost_per_kg   # per kg

h2_price_per_kg = 2.5

existing_pv_mw = 200                        # Existing PV capacity, model will optimize additional MW
capacity_requirement = 100                  # Existing capacity, # max of ST-2 and ST-3
capacity_credit_battery = 0.33

turbine_min_mw = 0
turbine_ramp_mw_per_min = 100               # unlimited

co2_emissions_lb_mmbtu = 117                # from "Capital Costs and Performance Characteristics for Utility Scale Power Generating Technologies" EIA, pg 28, #5

# efficiencies; blends of h2 and ng will have efficiencies that are the weighted average of the composition
h2_density = 2.41                           # kg per 1000 cf
ng_density = 20.16                          # kg per 1000 cf/mmbtu
h2_lhv = 33.391                             # kWh/kg
ng_lhv = 13.09                              # kWh/kg
h2_turb_conv = 0.39 * h2_lhv
ng_turb_conv = 0.33 * ng_lhv
mmbtu_to_ng_kg = 20.133                     # Based on density of 0.711 kg/m3

# example day of data
pv_cfs = np.sin(np.deg2rad(np.linspace(0, 180, 24)))
loads_mw = np.ones(24) * 100
reserve_mw = np.ones(24) * 100
prices_used = prices[:24]
ng_prices_per_mmbtu = np.ones(24) * 3

n_timesteps = len(pv_cfs)
timestep_hrs = 1
s_per_ts = timestep_hrs * 3600
pv_capacity_factors = {t:
                            {'pv_resource_config': {
                                'capacity_factor': 
                                    pv_cfs[t]}} for t in range(n_timesteps)}

re_h2_parameters = {
    "pv_mw": existing_pv_mw,
    "pv_mw_ub": 1000,
    "batt_mw": 0,
    "batt_mwh": 0,
    "pem_mw": 0,
    "pem_bar": pem_bar,
    "pem_temp": pem_temp,
    "h2_turb_conv": h2_turb_conv,
    "ng_turb_conv": ng_turb_conv,
    "tank_size": capacity_requirement * 1e3 / (h2_blend_ratio * h2_turb_conv + (1 - h2_blend_ratio) * ng_turb_conv),
    "turb_mw": capacity_requirement,
    "tank_holdup_init": 0,

    "pv_resource": pv_capacity_factors,
    "load": loads_mw,
    'reserve': reserve_mw,
    "LMP": prices_used,
    "NG_prices": ng_prices_per_mmbtu,

    "pv_cap_cost": wind_cap_cost,
    "pv_op_cost": wind_op_cost,
    "batt_cap_cost_kw": batt_cap_cost_kw,
    "batt_cap_cost_kwh": batt_cap_cost_kwh,
    "batt_rep_cost_kwh": batt_rep_cost_kwh,
    "pem_cap_cost": pem_cap_cost,
    "pem_op_cost": pem_op_cost,
    "pem_var_cost": pem_var_cost,
    "tank_cap_cost_per_kg": tank_cap_cost_per_kg,
    "tank_op_cost":tank_op_cost,
    "turbine_cap_cost": turbine_cap_cost,
    "turbine_op_cost": turbine_op_cost,
    "turbine_var_cost": turbine_var_cost,
    "turbine_min_mw": turbine_min_mw,
    "turbine_ramp_mw_per_min": turbine_ramp_mw_per_min,

    "h2_price_per_kg": h2_price_per_kg,

    "design_opt": True
}
