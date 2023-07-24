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
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.Simulation_Data_subscenario import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.NE_case_study.Train_NN_Surrogates_steady_state import TrainNNSurrogates
import pathlib
from dispatches_data.api import path

def main():
    # train static frequency surrogate model for NE case study
    # path_to_data_package is a standard pathlib.Path object
    path_to_data_package = path("dynamic_sweep")
    dispatch_data_path = path_to_data_package / "NE" / "Dispatch_data_NE_Dispatch_whole.csv"
    input_data_path = path_to_data_package / "NE" / "sweep_parameters_results_NE_whole.h5"
    case_type = 'NE'
    num_sims = 192

    # test TimeSeriesClustering
    print('Read simulation data')
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    
    # TrainNNSurrogates, dispatch cf
    print('Start train dispatch frequency surrogate')
    NNtrainer = TrainNNSurrogates(simulation_data)
    NN_model = NNtrainer.train_NN_cf([4,25,25,1])
    NN_frequency_model_path = str(pathlib.Path.cwd().joinpath(f'steady_state/tanh_25_25/NE_steady_state'))
    NN_frequency_param_path = str(pathlib.Path.cwd().joinpath(f'steady_state/tanh_25_25/NE_steady_state_params.json'))
    # NNtrainer.save_model(NN_model, NN_frequency_model_path, NN_frequency_param_path)
    NNtrainer.plot_R2_results(NN_frequency_model_path, NN_frequency_param_path, fig_name = f'NE_steady_state.jpg')



if __name__ == "__main__":
    main()