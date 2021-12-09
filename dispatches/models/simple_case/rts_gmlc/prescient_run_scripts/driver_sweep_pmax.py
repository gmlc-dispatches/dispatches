import mpi4py.MPI as mpi

rank = mpi.COMM_WORLD.Get_rank()
index = rank

start_date = '01-02-2020'
days = 364
base_output_dir = '/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/prescient_results/sensitivity_sweep_over_pmax'
def main():
    import prescient.scripts.simulator as simulator
    from prescient.scripts.runner import parse_line

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
    '--sced-solver=gurobi_direct',
    '--sced-solver-options="threads=4"',
    '--ruc-horizon=36',
    '--simulator-plugin=/home/jhjalvi/git/dispatches/dispatches/models/simple_case/rts_gmlc/prescient_run_scripts/plugin_sweep_pmax.py',
    '--disable-stackgraphs'
    ]

    
    ## redirect stdout to /dev/null
    import os
    import sys
    sys.stdout = open(os.devnull, 'w')
    simulator.main(args=options)
    sys.stdout = sys.__stdout__
    print("Index {} complete".format(index))

if __name__ == '__main__':
    main()
