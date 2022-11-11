#!/bin/bash
#SBATCH --nodes=1               # Number of nodes
#SBATCH --time=00-16:00:00      # Job should run for up to 12 hours
#SBATCH --account=gmihybridsys  # Where to charge NREL Hours
#SBATCH --mail-user=Bernard.Knueven@nrel.gov  # If you want email notifications
#SBATCH --mail-type=BEGIN,END,FAIL		 # When you want email notifications
#SBATCH --output=prescient_%j.out  # %j will be replaced with the job ID

## to run:
## sbatch --array=0-38 fossil_sweep_15_500.sh

JOBS_PER_NODE=25

module load conda
module load xpressmp

conda activate dispatches

python fossil_sweep_15_500.py $((SLURM_ARRAY_TASK_ID*JOBS_PER_NODE)) $(((SLURM_ARRAY_TASK_ID+1)*JOBS_PER_NODE))
