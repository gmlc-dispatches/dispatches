import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
from only_rev_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE
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

model = conceptual_design_dynamic_RE(default_input_params, num_rep_days = 3, verbose = False, plant_type = 'RE')
# print(value(model.pmax))

milp_solver = SolverFactory('cbc')

nlp_solver = SolverFactory('ipopt')
# nlp_solver.options['max_iter'] = 500
nlp_solver.solve(model, tee=True)
# dh = DegeneracyHunter(model, solver=milp_solver)
# dh.check_residuals(tol=1e-5)
# model.scenario_model_0.blocks[9].process.fs.windpower.electricity.pprint()
# model.scenario_model_0.blocks[9].process.fs.windpower.capacity_factor.pprint()
# model.scenario_model_0.blocks[9].process.fs.windpower.system_capacity.pprint()