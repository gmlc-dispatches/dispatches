#!/bin/bash
#SBATCH --nodes=1               # Number of nodes
#SBATCH --time=00-04:00:00      # Job should run for up to 12 hours
#SBATCH --account=gmihybridsys  # Where to charge NREL Hours
#SBATCH --mail-user=Bernard.Knueven@nrel.gov  # If you want email notifications
#SBATCH --mail-type=BEGIN,END,FAIL		 # When you want email notifications
#SBATCH --output=prescient_%j.out  # %j will be replaced with the job ID


module load conda
module load xpressmp

conda activate dispatches

python summarize_fossil_sweep.py 
