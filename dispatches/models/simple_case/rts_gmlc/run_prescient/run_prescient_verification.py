#NOTE: This uses the `naerm` branch of Prescient on the Sandia gitlab server
import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line

start_date = '01-02-2020'
days = 364
base_output_dir = '/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/run_prescient/prescient_verification_results'

options = [
'--data-directory=deterministic_scenarios',
'--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
'--output-directory='+base_output_dir,
'--run-deterministic-ruc',
'--start-date='+start_date,
'--num-days={}'.format(days),
'--sced-horizon=4',
'--traceback',
'--ruc-mipgap=0.001',
'--deterministic-ruc-solver=gurobi_direct',
'--deterministic-ruc-solver-options="threads=4"',
'--sced-solver=gurobi_direct',
'--sced-solver-options="threads=4"',
'--ruc-horizon=36',
'--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/run_prescient/verify_surrogate_plugin.py',
'--disable-stackgraphs'
]

simulator.main(args=options)
