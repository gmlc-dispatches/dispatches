#Test script to make sure lmp cap works in Xian's analysis code
#Compare revenues between case with and without lmp cap

import pandas as pd
import os
import sys
from prescient_analysis_code import PrescientSimulationData

#Location of raw Prescient output
result_data_dir = '../run_prescient/basecase_prescient_runs/deterministic_simulation_naerm_branch_basecase/'

#Drop results into this directory
summary_dir = 'summary_results/'

sim = PrescientSimulationData(result_data_dir = result_data_dir, network_data_dir = '/home/jhjalvi/git/RTS-GMLC/RTS_Data/SourceData/',
custom_string = '',custom_string2 = '')
sim.read_result_files()

# summary dataframe
df1 = sim.summarize_results(include_generator_param = True,cap_lmp = 200.0)
df2 = sim.summarize_results(include_generator_param = True,cap_lmp = None)

# write the summary to csv
#df.to_csv(summary_dir + 'result_index_{}.csv'.format(result_idx),index = False)
