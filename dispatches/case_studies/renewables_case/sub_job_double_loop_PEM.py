import os

this_file_path = os.path.dirname(os.path.realpath(__file__))


def submit_job(
    sim_id,
    wind_pmax,
    pem_pmax,
    pem_bid,
    reserve_factor,
    shortfall
):

    # create a directory to save job scripts
    job_scripts_dir = os.path.join(this_file_path, "sim_job_scripts")
    if not os.path.isdir(job_scripts_dir):
        os.mkdir(job_scripts_dir)

    file_name = os.path.join(job_scripts_dir, f"Benchmark_wind_pem_parameterized_rf_15_shortfall_200.sh")
    with open(file_name, "w") as f:
        f.write(
            "#!/bin/bash\n"
            + "#$ -M xchen24@nd.edu\n"
            + "#$ -m ae\n"
            + "#$ -q long\n"
            + f"#$ -N re-wind-pem-sim_{sim_id}\n"
            + "conda activate regen\n"
            + "export LD_LIBRARY_PATH=~/.conda/envs/regen/lib:$LD_LIBRARY_PATH \n"
            + "module load ipopt/3.14.2 \n"
            + "module load ompi/3.0.0/intel/18.0 \n"
            + "module load intel/18.0 \n"
            + "module load gurobi/9.5.1\n"
            + f"python ./run_double_loop_PEM.py --sim_id {sim_id} --wind_pmax {wind_pmax} --pem_pmax {pem_pmax} --pem_bid {pem_bid} --reserve_factor {reserve_factor} --shortfall {shortfall}"
        )

    os.system(f"qsub {file_name}")


sim_id = 1
wind_pmax = 847
pem_pmax = 200
pem_bid = 25
reserve_factor =0.15
shortfall = 200

submit_job(sim_id, wind_pmax, pem_pmax, pem_bid, reserve_factor, shortfall)
