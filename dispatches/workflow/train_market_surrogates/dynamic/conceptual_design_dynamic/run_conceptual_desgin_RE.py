import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from conceptual_design_dynamic import conceptual_design_dynamic_RE
# wind_resource = 

default_input_params = {
    "wind_mw": 847,
    "wind_mw_ub": 10000,
    "batt_mw": 4874,
    "pem_mw": None,
    "pem_bar": None,
    "pem_temp": None,
    "tank_size": None,
    "tank_type": None,
    "turb_mw": None,

    "wind_resource": None,
    "h2_price_per_kg": None,
    "DA_LMPs": None,

    "design_opt": True,
    "extant_wind": True
} 

model = conceptual_design_dynamic_RE(default_input_params, num_rep_days = 32, verbose = False, plant_type = 'RE')