import sys
sys.path.append("../")
# Import Pyomo libraries
from pyomo.environ import value
# from pyomo.util.infeasible import log_close_to_bounds

# from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver

from stochastic_surrogate import stochastic_surrogate_optimization_problem
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter

# Inputs for stochastic problem
capital_payment_years = 3
plant_lifetime = 20
heat_recovery = True
p_upper_bound = 300


build_tic = perf_counter()
m = stochastic_optimization_problem(
    heat_recovery=heat_recovery,
    capital_payment_years=capital_payment_years,
    p_upper_bound=p_upper_bound,
    plant_lifetime=20,
    power_demand=power_demand,
    lmp=lmp_scenarios.tolist(),
    lmp_weights=lmp_weights.tolist())
build_toc = perf_counter()

solver = get_solver()
solver.options = {
    "tol": 1e-6
}
solver.solve(m, tee=True)
