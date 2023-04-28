#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################

# This script requires steady-state-sweep-data package. To install the data package
# run `pip install git+https://github.com/gmlc-dispatches/steady-state-sample-data.git`
# Please see https://github.com/gmlc-dispatches/data-packages for instructions.

import pandas as pd
from dispatches_data.api import path
from sklearn.model_selection import train_test_split
from dispatches.util import train_surrogate_model

"""
Train a surrogate model for the number of startups
"""
# Load prescient data
f_inputs = path("steady_state_sweep") / "prescient_generator_inputs.h5"
df_inputs = pd.read_hdf(f_inputs)

f_startups = path("steady_state_sweep") / "prescient_generator_startups.h5"
df_nstartups = pd.read_hdf(f_startups)

x = df_inputs.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9]].to_numpy()
z = df_nstartups["# Startups"].to_numpy()

# 1/3 of data is used for test
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

# Construct Scikit surrogate model for nstartups
train_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="scikit",
    activation="tanh",
    hidden_layers=(100, 50),
    filename="nstartups",
)


"""
Train a surrogate model for the revenue
"""
# Load prescient data
f_inputs = path("steady_state_sweep") / "prescient_generator_inputs.h5"
df_inputs = pd.read_hdf(f_inputs)

f_outputs = path("steady_state_sweep") / "prescient_generator_outputs.h5"
df_outputs = pd.read_hdf(f_outputs)

x = df_inputs.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9]].to_numpy()
z = df_outputs["Total Revenue [$]"].to_numpy()

# 1/3 of data is used for test
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

# Construct a scikit surrogate model for revenue
train_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="scikit",
    activation="tanh",
    hidden_layers=(100, 50),
    filename="revenue",
)


"""
Train a surrogate model for zones
"""
# Load prescient data
f_inputs = path("steady_state_sweep") / "prescient_generator_inputs.h5"
df_inputs = pd.read_hdf(f_inputs)

f_zones = path("steady_state_sweep") / "prescient_generator_zones.h5"
df_zones = pd.read_hdf(f_zones)

x = df_inputs.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9]].to_numpy()
z = df_zones.iloc[:, 1:12].to_numpy()

# 1/3 of data is used for test
X_train, X_test, z_train, z_test = train_test_split(x, z, test_size=0.33, random_state=42)

# Construct a scikit surrogate model for zones
train_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="scikit",
    activation="tanh",
    hidden_layers=(100, 100),
    filename="zones",
)