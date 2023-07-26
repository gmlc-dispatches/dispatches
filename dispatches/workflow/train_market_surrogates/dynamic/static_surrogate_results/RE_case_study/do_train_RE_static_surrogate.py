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
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.RE_case_study.Train_static_clustering_NN_surrogate import TrainNNSurrogates
from dispatches.workflow.train_market_surrogates.dynamic.static_surrogate_results.RE_case_study.clustering_dispatch_wind_pem_static import ClusteringDispatchWind
import pathlib
from dispatches_data.api import path

def main():
    num_sims = 224
    num_clusters = 20
    case_type = 'RE'
    wind_gen = '303_WIND_1'
    # train static frequency surrogate model for NE case study
    # path_to_data_package is a standard pathlib.Path object
    path_to_data_package = path("dynamic_sweep")
    dispatch_data_path = path_to_data_package / "RE" / "Dispatch_data_RE_H2_Dispatch_whole.csv"
    input_data_path = path_to_data_package / "RE" / "sweep_parameters_results_RE_H2_whole.h5"
    wind_data_path = path_to_data_package / "RE" / "Real_Time_wind_hourly.csv"
    clustering_model_path = 'static_clustering_wind_pmax.pkl'
    
    clustering_class = ClusteringDispatchWind(dispatch_data_path, wind_data_path, input_data_path, wind_gen, num_clusters)
    simulation_data = SimulationData(dispatch_data_path, input_data_path, num_sims, case_type)
    NNtrainer = TrainNNSurrogates(simulation_data, clustering_class, clustering_model_path)

    model = NNtrainer.train_NN_frequency([4,45,75,45,num_clusters])
    NN_model_path = f'static_surrogate/ss_surrogate_model_wind_pmax'
    NN_param_path = f'static/ss_surrogate_param_wind_pmax.json'
    # NNtrainer.save_model(model,NN_model_path,NN_param_path)
    NNtrainer.plot_R2_results(NN_model_path, NN_param_path)


if __name__ == '__main__':
    main()