#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################

'''
Test bugs, check input data...
'''

from pyomo.common.fileutils import this_file_dir
import sys, os, json
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


# use renewable energy codes in 'RE_flowsheet.py'
# import specified functions instead of using *
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

# sys.path.append(os.path.join(this_file_dir(),"../../../../../"))
from dispatches.case_studies.renewables_case.RE_flowsheet import add_wind, add_battery, \
    create_model 

from pyomo.environ import ConcreteModel, SolverFactory, units, Var, \
    TransformationFactory, value, Block, Expression, Constraint, Param, \
    Objective, NonNegativeReals, ConstraintList, Set, units as pyunits, RangeSet
from pyomo.network import Arc
from pyomo.util.infeasible import log_close_to_bounds

# Import IDAES components
from idaes.core import FlowsheetBlock, UnitModelBlockData

# Import heat exchanger unit model
from idaes.models.unit_models.heater import Heater
from idaes.models.unit_models.pressure_changer import PressureChanger
from idaes.models.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.models_extra.power_generation.costing.power_plant_costing import get_PP_costing

# Import steam property package
from idaes.models.properties.iapws95 import htpx, Iapws95ParameterBlock
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.core.solvers.get_solver import get_solver
import idaes.logger as idaeslog
import pyomo.environ as pyo

# from read_scikit_to_omlt import load_scikit_mlp
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#omlt can encode the neural networks in Pyomo
import omlt
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation
from omlt.io import load_keras_sequential
from PySAM.ResourceTools import SRW_to_wind_data



# import codes from Darice
from idaes.apps.grid_integration.multiperiod.multiperiod import MultiPeriodModel

# from Darice's codes import functions to build the multi period model
from dispatches.case_studies.renewables_case.wind_battery_LMP import wind_battery_variable_pairs, \
                                wind_battery_periodic_variable_pairs, wind_battery_om_costs, \
                                initialize_mp, wind_battery_model, wind_battery_mp_block

# read the default wind speed data
wind_data_path = os.path.join(this_file_dir(),'../../../../case_studies/renewables_case/data/44.21_-101.94_windtoolkit_2012_60min_80m.srw')
wind_data = SRW_to_wind_data(wind_data_path)

# pick up a default wind speed data
wind_speeds = [wind_data['data'][i][2] for i in range(8736)]

print(len((wind_speeds)))
print(np.mean(wind_speeds))
print(wind_speeds[0:24])
