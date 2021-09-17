import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line

start_date = '01-02-2020'
days = 364
base_output_dir = '/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/'
index = 0

options = [
#'--data-directory=/home/jhjalvi/git/idaes_parameter_sweep/rts_gmlc_data/deterministic_scenarios',
#'--model-directory=..|..|..|software|prescient|prescient|models|knueven',
'--data-directory=deterministic_scenarios',
'--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
'--output-directory='+base_output_dir+'deterministic_simulation_output_index_{}'.format(index),
'--run-deterministic-ruc',
'--start-date='+start_date,
'--num-days={}'.format(days),
'--sced-horizon=4',
'--traceback',
'--ruc-mipgap=0.001',
#'--deterministic-ruc-solver=gurobi_direct',
#'--deterministic-ruc-solver-options="threads=1"',
'--deterministic-ruc-solver=cbc',
'--deterministic-ruc-solver-options="feas=off cuts=off GMI=on mixed=on probing=on two=on DivingP=on DivingS=on zeroT=1e-10 primalT=1e-6 dualT=1e-6"',
#'--sced-solver=gurobi_direct',
#'--sced-solver-options="threads=1"',
#'--sced-solver-options="zeroT=1e-10 primalT=1e-6 dualT=1e-6"',
'--ruc-horizon=36',
'--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/verify_surrogate.py'
#'--disable-stackgraphs',
]

simulator.main(args=options)

# def get_subprocess_options(options_strings):
#     options = []
#     for opt_str in options_strings:
#         options.extend(parse_line(opt_str))
#     return options

# options = get_subprocess_options(options)






