#NOTE: This uses the `naerm` branch of Prescient on the Sandia gitlab server
import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line

start_date = '01-02-2020'
days = 364
base_output_dir = '/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/run_prescient'
index = 0

options = [
'--data-directory=deterministic_scenarios',
'--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
'--output-directory='+base_output_dir+'deterministic_simulation_output_index_{}'.format(index),
'--run-deterministic-ruc',
'--start-date='+start_date,
'--num-days={}'.format(days),
'--sced-horizon=4',
'--traceback',
'--ruc-mipgap=0.001',
'--deterministic-ruc-solver=gurobi_direct',
'--deterministic-ruc-solver-options="threads=4"',
#'--deterministic-ruc-solver=cbc',
#'--deterministic-ruc-solver-options="feas=off cuts=off GMI=on mixed=on probing=on two=on DivingP=on DivingS=on zeroT=1e-10 primalT=1e-6 dualT=1e-6"',
'--sced-solver=gurobi_direct',
'--sced-solver-options="threads=4"',
#'--sced-solver-options="zeroT=1e-10 primalT=1e-6 dualT=1e-6"',
'--ruc-horizon=36',
'--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/run_prescient/basecase_gen_plugin.py',
#'--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/verify_surrogate.py',
'--disable-stackgraphs'
]

simulator.main(args=options)





