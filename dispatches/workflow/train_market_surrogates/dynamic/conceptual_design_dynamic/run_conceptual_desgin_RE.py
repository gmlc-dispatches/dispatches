#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2021 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np
# from rev_nstartups_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE, record_result
from dispatches.workflow.train_market_surrogates.dynamic.conceptual_design_dynamicnew_full_surrogate_omlt_v1_conceptual_design_dynamic import conceptual_design_dynamic_RE, record_result
from pyomo.environ import value, SolverFactory
from idaes.core.util.model_diagnostics import DegeneracyHunter
import time
import os
from pyomo.common.fileutils import this_file_dir
from idaes.core.util import to_json, from_json


default_input_params = {
    "wind_mw": 440.5,
    "wind_mw_ub": 10000,
    "batt_mw": 40.05,
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
    "extant_wind": False
} 


start_time = time.time()
model = conceptual_design_dynamic_RE(default_input_params, num_rep_days = 32, verbose = False, plant_type = 'RE')

nlp_solver = SolverFactory('ipopt')
# nlp_solver.options['max_iter'] = 500
nlp_solver.options['acceptable_tol'] = 1e-8
nlp_solver.solve(model, tee=True)
end_time = time.time()

print('------------------------------------------------------------------------')
print('Time for solving the model is {} seconds'.format(end_time - start_time))
print('------------------------------------------------------------------------')

record_result(model,32)

to_json(model, fname = 'run3.json.gz', human_read = True)
# print(model.scenario_model_1.pprint())
# print(default_input_params['wind_resource'])
# milp_solver = SolverFactory('cbc')
# dh = DegeneracyHunter(model, solver=milp_solver)
# dh.check_residuals(tol=1e-5)
