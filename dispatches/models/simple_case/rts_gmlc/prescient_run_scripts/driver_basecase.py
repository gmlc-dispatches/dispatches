from pyomo.common.fileutils import this_file_dir
import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line
import os

start_date = '01-02-2020'
days = 364
base_index = 0
base_output_dir = '/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/prescient_results/basecase_runs/run_{}'.format(base_index)
base_output_dir = this_file_dir()+'/../prescient_results/basecase_runs/run_{}'.format(base_index)
os.makedirs(base_output_dir, exist_ok=True)

def main():
    import prescient.scripts.simulator as simulator
    from prescient.scripts.runner import parse_line

    options = [
    '--data-directory=deterministic_scenarios',
    '--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
    '--output-directory='+base_output_dir+'/deterministic_simulation_output_index_{}'.format(index),
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
	'--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/prescient_run_scripts/plugin_basecase_generator.py',
    '--disable-stackgraphs'
    ]

    
    ## redirect stdout to /dev/null
    #import os
    # import sys
    # sys.stdout = open(os.devnull, 'w')
    # print("Running Index".format(index))
    simulator.main(args=options)
    # sys.stdout = sys.__stdout__
    # print("Index {} complete".format(index))

if __name__ == '__main__':
    main()



