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

#NOTE: This uses the `naerm` branch of Prescient on the Sandia gitlab server
import prescient.scripts.simulator as simulator
from prescient.scripts.runner import parse_line
from pyomo.common.fileutils import this_file_dir
import os

start_date = '01-02-2020'
days = 364
base_index = 2
base_output_dir = os.join(this_file_dir(),'/../prescient_results/verification_runs/scikit_run_{}'.format(base_index))
os.makedirs(base_output_dir, exist_ok=True)

#location of surrogate plugin
surrogate_plugin = os.join(this_file_dir(),"plugin_verify_surrogate.py"

def main():
    import prescient.scripts.simulator as simulator
    from prescient.scripts.runner import parse_line

    options = [
    '--data-directory=deterministic_scenarios',
    '--model-directory=/home/jhjalvi/git/prescient_idaes/prescient/models/knueven',
    '--output-directory='+base_output_dir+'/deterministic_simulation_output_index_{}'.format(base_index),
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
	'--simulator-plugin={}'.format(surrogate_plugin),
    '--disable-stackgraphs'
    ]

    simulator.main(args=options)

if __name__ == '__main__':
    main()
