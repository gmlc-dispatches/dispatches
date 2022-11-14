import pytest
from pyomo.common import unittest as pyo_unittest
from dispatches.workflow.train_market_surrogates.dynamic.Simulation_Data import SimulationData
from dispatches.workflow.train_market_surrogates.dynamic.Time_Series_Clustering import TimeSeriesClustering
import idaes.logger as idaeslog
import os
import numpy as np

test_simultaion_data = 'test_data/simdatatest.csv'
test_input_data = 'test_data/inputdatatest.h5'
num_sims = 3
case_type = 'NE'
fixed_pmax = True

sd = SimulationData(test_simultaion_data, test_input_data, num_sims, case_type, fixed_pmax)
dispatch_array, index = sd._read_data_to_array()


num_clusters = 1
tsc = TimeSeriesClustering(num_clusters, sd)

model = tsc.clustering_data()
tsc.save_clustering_model(model,'test_clustering_model.json')