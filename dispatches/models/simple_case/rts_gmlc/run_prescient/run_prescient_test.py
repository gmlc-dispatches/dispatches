#script to simply check whether Prescient is running correctly

import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line

index = 0
start_date = '01-02-2020'
days = 1

options = [
'--data-directory=deterministic_scenarios',
#'--model-directory=..|..|..|software|prescient|prescient|models|knueven',
'--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
'--output-directory=deterministic_simulation_output_index_{}'.format(index),
'--run-deterministic-ruc',
'--start-date='+start_date,
'--num-days={}'.format(days),
'--sced-horizon=4',
'--traceback',
'--ruc-mipgap=0.001',
'--deterministic-ruc-solver=gurobi_direct'
'--deterministic-ruc-solver-options="threads=1"',
'--sced-solver=gurobi_direct',
'--sced-solver-options="threads=1"',
'--ruc-horizon=36',
'--simulator-plugin=sweep_plugin.py',
'--disable-stackgraphs',
]

simulator.main(args=options)

