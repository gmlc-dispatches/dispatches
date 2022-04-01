import json
import os
import sys
from timeit import default_timer
import pandas as pd
from pyomo.environ import TerminationCondition
from pyomo.common.tempfiles import TempfileManager
# TempfileManager.tempdir = "/tmp/scratch"

from load_parameters import prices_used
from wind_battery_PEM_LMP import wind_battery_pem_optimize
from wind_battery_PEM_tank_turbine_LMP import wind_battery_pem_tank_turb_optimize


solver = "glpk"

if len(sys.argv) < 2:
    raise RuntimeError("Specify which model to run: 'bp' or 'bptt'")
else:
    if sys.argv[1] == 'bp':
        func = wind_battery_pem_optimize
        outname = f"wind_battery_PEM_LMP_timehorz_{solver}.json"
    elif sys.argv[1] == 'bptt':
        func = wind_battery_pem_tank_turb_optimize
        outname = f"wind_battery_PEM_tank_turbine_LMP_timehorz_{solver}.json"
    else:
        raise ValueError

try:
    raise ValueError
except Exception as e:
    print(e)


h2_prices = [2.0, 2.5, 3.0]
year_intervals = [52, 24, 12, 4, 2, 1]
# year_intervals = [52]
i = 0

if os.path.exists(outname):
    wind_battery_pem_res = pd.read_json(outname, orient='index', convert_axes=False)
    res = json.loads(wind_battery_pem_res.to_json(orient='index'))
else:
    res = {}


for h2_price in h2_prices:
    for y in year_intervals:
        if str(i) in res.keys():
            if res[str(i)]["glpk_solved"]:
                i += 1
                continue
        n_hours = min(int(8760 / y), len(prices_used) - 1)
        print("Nhours", n_hours)
        start = default_timer()
        design_opt, solver_res, model_create_time = func(n_hours, h2_price)
        total_run_time = default_timer() - start
        design_opt.update({
            'h2_price': h2_price,
            "ts": n_hours,
            "model_creation": model_create_time,
            "total_run_time": total_run_time
        }
        )
        if solver == "ipopt":
            design_opt.update({
                "model_vars": solver_res[0]['Problem'][0]['Number of variables'],
                "ipopt_solved": solver_res[1],
                "ipopt_its": solver_res[2],
                "ipopt_time": solver_res[3],
                "ipopt_reg": solver_res[4]
            })
        elif solver == "glpk":
            design_opt.update({
                "glpk_solved": solver_res.solver.termination_condition == TerminationCondition.optimal,
                "glpk_time": solver_res.solver.time
            })
        res[str(i)] = design_opt
        with open(outname, 'w') as f:
            json.dump(res, f)
        i += 1
