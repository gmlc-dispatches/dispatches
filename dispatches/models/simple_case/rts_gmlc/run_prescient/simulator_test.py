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
'--deterministic-ruc-solver=gurobi',#_direct'
'--deterministic-ruc-solver-options="threads=1"',
#'--deterministic-ruc-solver=cbc',
#'--deterministic-ruc-solver-options="feas=off cuts=off GMI=on mixed=on probing=on two=on DivingP=on DivingS=on zeroT=1e-10 primalT=1e-6 dualT=1e-6"',
'--sced-solver=gurobi',#_direct',
'--sced-solver-options="threads=1"',
# '--sced-solver-options="zeroT=1e-10 primalT=1e-6 dualT=1e-6"',
'--ruc-horizon=36',
#'--simulator-plugin=sweep_plugin.py',
#'--disable-stackgraphs',
]

simulator.main(args=options)

