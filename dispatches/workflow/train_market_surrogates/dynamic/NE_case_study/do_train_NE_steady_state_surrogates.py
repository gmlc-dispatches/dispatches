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

import os
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.NE_case_study.Train_NN_Surrogates_steady_state import TrainNNSurrogates
import pathlib

def main():
    # for NE case study
    dispatch_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','..','datasets','results_nuclear_sweep','Dispatch_data_NE_whole.csv'))
    input_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','..','datasets','results_nuclear_sweep','sweep_parameters_results_NE_whole.h5'))
    case_type = 'NE'
    num_clusters = 30
    num_sims = 192
    input_layer_node = 4
    filter_opt = True

    # for FE case study
    # dispatch_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_fossil_sweep_revised_fixed_commitment','Dispatch_data_FE_Dispatch_whole.csv'))
    # input_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_fossil_sweep_revised_fixed_commitment','sweep_parameters_results_FE_whole.h5'))
    # case_type = 'FE'
    # num_clusters = 20
    # num_sims = 400
    # input_layer_node = 4
    # filter_opt = True

    # for RE case study
    # dispatch_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_renewable_sweep_Wind_H2','Dispatch_data_RE_H2_Dispatch_whole.csv'))
    # input_data_path = str(pathlib.Path.cwd().joinpath('..','..','..','..','..','datasets','results_renewable_sweep_Wind_H2','sweep_parameters_results_RE_H2_whole.h5'))
    # case_type = 'RE'
    # num_clusters = 20
    # num_sims = 224
    # input_layer_node = 4
    # filter_opt = False

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    

    # TrainNNSurrogates, dispatch cf
    print('Start train dispatch frequency surrogate')
    NNtrainer = TrainNNSurrogates(simulation_data)
    NN_model = NNtrainer.train_NN_cf([input_layer_node,75,1])
    NN_frequency_model_path = str(pathlib.Path.cwd().joinpath(f'NE_steady_state'))
    NN_frequency_param_path = str(pathlib.Path.cwd().joinpath(f'NE_steady_state_params.json'))
    NNtrainer.save_model(NN_model, NN_frequency_model_path, NN_frequency_param_path)
    NNtrainer.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'NE_steady_state.jpg')



if __name__ == "__main__":
    main()