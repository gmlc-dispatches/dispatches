import os

this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(
    sim_id,
    wind_pmax,
    battery_energy_capacity,
    battery_pmax,
    n_scenario,
    participation_mode,
    reserve_factor,
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"benchmark_wind_battery_sim_rf_{reserve_factor}_shortfall_500.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N re-wind-battery-sim_{sim_id}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load gurobi/9.5.1\n"
            + "module load ipopt/3.14.2 \n"
            + f"python ./run_double_loop_battery.py --sim_id {sim_id} --wind_pmax {wind_pmax} --battery_energy_capacity {battery_energy_capacity} --battery_pmax {battery_pmax} --n_scenario {n_scenario} --participation_mode {participation_mode} --reserve_factor {reserve_factor}"
        )

    os.system(f"qsub {file_name}")


if __name__ == "__main__":

    sim_id = 0

    wind_pmax = 847

    battery_energy_capacity = 200

    battery_pmax = 50

    n_scenario = 10

    participation_mode = 'Bid'

    reserve_factor = 0.15

    submit_job(sim_id, wind_pmax, battery_energy_capacity, battery_pmax, n_scenario, participation_mode, reserve_factor)
    
