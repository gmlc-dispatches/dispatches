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

import numpy as np
import pandas as pd
import os
import pickle
import json
from idaes.core.surrogate.alamopy import AlamoTrainer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
import tensorflow as tf
tf.keras.backend.set_floatx('float64')


def train_surrogate_alamo(
    x_train,
    z_train,
    x_labels,
    z_labels,
    config,
    file_dir="",
    filename="surrogate",
):
    # Determine the range of each input
    num_inputs = x_train.shape[1]
    x_min = list(x_train.min(axis=0))
    x_max = list(x_train.max(axis=0))

    # Store the data in a DataFrame
    data_in = pd.DataFrame(x_train, columns=x_labels)
    data_out = pd.DataFrame(z_train, columns=z_labels)
    data = pd.concat([data_in, data_out], axis=1)

    input_bounds = {x_labels[i]: (x_min[i], x_max[i]) for i in range(num_inputs)}
    config["filename"] = file_dir + filename + "_run.alm"
    config["overwrite_files"] = True

    trainer = AlamoTrainer(
        input_labels=x_labels,
        output_labels=z_labels,
        input_bounds=input_bounds,
        training_dataframe=data,
    )

    for opt, val in config.items():
        setattr(trainer.config, opt, val)

    # Train the surrogate model
    success, alm_surr, msg = trainer.train_surrogate()

    # Save the surrogate to file
    alm_surr.save_to_file(file_dir + filename + ".json", overwrite=True)

    return success, alm_surr, msg


def train_surrogate_keras(
    x_train,
    z_train,
    hidden_layers=(100, 50),
    activation="tanh",
    filename="surrogate",
):
    num_inputs = x_train.shape[1]
    num_outputs = (z_train.shape[1] if len(z_train.shape) == 2 else 1)

    model = Sequential(name=filename)
    model.add(Input(num_inputs))

    for nodes in hidden_layers:
        model.add(Dense(nodes, activation=activation))
    
    model.add(Dense(num_outputs))
    model.compile(optimizer=Adam(), loss='mse')
    history = model.fit(x=x_train, y=z_train, verbose=1, epochs=500)

    model.save("keras_models/" + filename)


def train_surrogate_scikit(
    x_train,
    z_train,
    activation="tanh",
    hidden_layers=(100, 50),
    filename="surrogate",
):
    # train scikit MLP Regressor model
    print("Training NN model ...")
    model = MLPRegressor(
        activation=activation,
        hidden_layer_sizes=hidden_layers,
    ).fit(x_train, z_train)

    # check the cross validation scores
    # print("Running Cross Validation Score...")
    # scores = cross_val_score(model, x_train, z_train, cv=5)

    # save model to pickle
    print("Saving model ...")
    with open("scikit_models/" + filename + ".pkl", 'wb') as f:
        pickle.dump(model, f)


def train_surrogate_model(
    x_data,
    z_data,
    x_labels=None,
    z_labels=None,
    trainer="alamo",
    shift_scale_data=True,
    alamo_config={"constant": True, "linfcns": True},
    activation="tanh",
    hidden_layers=(100, 50),
    filename="surrogate",
):
    """Build surrogate model for the given input-output data.

    Args:
        x_data: 
        z_data:
        x_label:
        z_label:
        trainer:
        shift_scale_data
    """

    if trainer not in ["alamo", "keras", "scikit"]:
        raise Exception(f"Unrecognized trainer {trainer}; Choose from [alamo, keras, scikit]")

    num_inputs = x_data.shape[1]
    num_outputs = (1 if len(z_data.shape) == 1 else z_data.shape[1])

    # Create default lables if labels are not provided
    if x_labels is None:
        x_labels = ["x" + str(i) for i in range(1, num_inputs + 1)]

    if z_labels is None:
        z_labels = ["z" + str(i) for i in range(1, num_outputs + 1)]

    # save scaling and training bounds
    xm = np.mean(x_data, axis=0)
    xstd = np.std(x_data, axis=0)
    zm = np.mean(z_data, axis=0)
    zstd = np.std(z_data, axis=0)
    xmin = list(np.min(x_data, axis=0))
    xmax = list(np.max(x_data, axis=0))
    data = {
        "xm_inputs": list(xm), 
        "xstd_inputs": list(xstd),
        "xmin": xmin,
        "xmax": xmax,
        "zm": list(zm) if hasattr(zm, "__iter__") else zm, 
        "zstd": list(zstd) if hasattr(zstd, "__iter__") else zstd,
    }

    cwd = os.getcwd()
    with open(cwd + "\\training_parameters_" + filename + ".json", 'w') as outfile:
        json.dump(data, outfile)

    # Shift the data by its mean value and scale it by its standard deviation
    if shift_scale_data:
        x_train_scaled = (x_data - xm) / xstd
        z_train_scaled = (z_data - zm) / zstd

    else:
        x_train_scaled = x_data
        z_train_scaled = z_data

    if trainer == "alamo":
        # Create a directory to store alamo models
        alm_dir = cwd + "\\alamo_models\\"
        if not os.path.exists(alm_dir):
            os.mkdir(os.path.join(cwd, "alamo_models"))

        success, alm_surr, msg = train_surrogate_alamo(
            x_train=x_train_scaled,
            z_train=z_train_scaled,
            x_labels=x_labels,
            z_labels=z_labels,
            config=alamo_config,
            file_dir=alm_dir,
            filename=filename,
        )

        return alm_surr

    elif trainer == "keras":
        # Create a directory to store keras models
        krs_dir = cwd + "\\keras_models\\"
        if not os.path.exists(krs_dir):
            os.mkdir(os.path.join(cwd, "keras_models"))

        train_surrogate_keras(
            x_train=x_train_scaled,
            z_train=z_train_scaled,
            activation=activation,
            hidden_layers=hidden_layers,
            filename=filename,
        )

    elif trainer == "scikit":
        # Create a directory to store keras models
        krs_dir = cwd + "\\scikit_models\\"
        if not os.path.exists(krs_dir):
            os.mkdir(os.path.join(cwd, "scikit_models"))

        train_surrogate_scikit(
            x_train=x_train_scaled,
            z_train=z_train_scaled,
            activation=activation,
            hidden_layers=hidden_layers,
            filename=filename,
        )
