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

this_file_path = os.path.dirname(os.path.realpath(__file__))

# submit double loop sweep jobs for wind battery case study using stochastic bidder.  

def submit_job(
    sim_id,
    wind_pmax,
    battery_energy_capacity,
    battery_pmax,
    n_scenario,
    participation_mode,
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"new_re_wind_battery_sweep_sb_sim_{sim_id}.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N new_re_wind_battery_sb_sweep_sim_{sim_id}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./run_double_loop_battery_new_setting.py --sim_id {sim_id} --wind_pmax {wind_pmax} --battery_energy_capacity {battery_energy_capacity} --battery_pmax {battery_pmax} --n_scenario {n_scenario} --participation_mode {participation_mode}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":
    
    from itertools import product

    sim_id = 0

    wind_pmax_list = list(range(50, 900, 50))
    
    # pmax_ratio: battery_power_pmax/wind_pmax
    pmax_ratio_list = [r/10 for r in range(1, 11, 1)]

    # battery size in hour
    battery_size = 4

    n_scenario_list = [10]
    participation_modes = ["Bid"]
    # "SelfSchedule"

    spec_comb_product = product(wind_pmax_list, pmax_ratio_list, n_scenario_list, participation_modes)

    for wind_pmax, p_max_ratio, n_scenario, pm in spec_comb_product:

        battery_pmax = wind_pmax * p_max_ratio
        battery_energy_capacity = battery_pmax * battery_size

        submit_job(
            sim_id=sim_id,
            wind_pmax=wind_pmax,
            battery_energy_capacity=battery_energy_capacity,
            battery_pmax=battery_pmax,
            n_scenario=n_scenario,
            participation_mode=pm,
            )

        sim_id += 1
