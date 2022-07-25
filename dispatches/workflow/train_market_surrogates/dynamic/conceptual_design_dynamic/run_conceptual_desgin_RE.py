import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from rev_nstartups_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE
from pyomo.environ import value, SolverFactory
from idaes.core.util.model_diagnostics import DegeneracyHunter

default_input_params = {
    "wind_mw": 177.5,
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

import time

start_time = time.time()
model = conceptual_design_dynamic_RE(default_input_params, num_rep_days = 20, verbose = False, plant_type = 'NU')

nlp_solver = SolverFactory('ipopt')
# nlp_solver.options['max_iter'] = 500
nlp_solver.options['acceptable_tol'] = 1e-8
nlp_solver.solve(model, tee=True)
end_time = time.time()

print('----------------------')
print('Time for solving the model is {} seconds'.format(end_time - start_time))
# milp_solver = SolverFactory('cbc')
# dh = DegeneracyHunter(model, solver=milp_solver)
# dh.check_residuals(tol=1e-5)
