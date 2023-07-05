#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
import os
from prescient_options import reserve_factor, shortfall, real_time_horizon

this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(
    sim_id,
    wind_pmax,
    battery_energy_capacity,
    battery_pmax,
    storage_bid,
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"new_Benchmark_wind_battery_parameterized_bidder_fix_commitment_rf_{int(reserve_factor * 1e2)}_shortfall_{shortfall}_rth_{real_time_horizon}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N new_Benchmark_re-wind-battery-pb-fix-commit-rf_{int(reserve_factor*100)}_shortfall_{shortfall}_rth_{real_time_horizon}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./run_double_loop_battery_parametrized.py --sim_id {sim_id} --wind_pmax {wind_pmax} --battery_energy_capacity {battery_energy_capacity} --battery_pmax {battery_pmax} --storage_bid {storage_bid}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":

    sim_id = 0

    wind_pmax = 847

    battery_energy_capacity = 200

    battery_pmax = 50

    storage_bid = 35

    submit_job(sim_id, wind_pmax, battery_energy_capacity, battery_pmax, storage_bid)
    
