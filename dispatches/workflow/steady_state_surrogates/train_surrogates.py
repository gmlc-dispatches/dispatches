import pandas as pd
from importlib import resources
from pathlib import Path
from sklearn.model_selection import train_test_split
from dispatches.util import build_surrogate_model


def get_data(filename):
    """Find the data file and read the data
    """
    with resources.path("dispatches.util.prescient_data", filename) as p:
        path_to_file = Path(p).resolve()

    return pd.read_hdf(path_to_file)

# Load prescient data
df_inputs = get_data("prescient_generator_inputs.h5")
df_nstartups = get_data("prescient_generator_startups.h5")
df_outputs = get_data("prescient_generator_outputs.h5")
df_dispatch_zones = get_data("prescient_generator_zones.h5")

# Extract the inputs and outputs as numpy arrays
x = df_inputs.iloc[:, [1, 2, 3, 4, 5, 6, 7, 9]].to_numpy()
z_nstartups = df_nstartups["# Startups"].to_numpy()
z_revenue = df_outputs["Total Revenue [$]"].to_numpy() / 1e6 # scale dollars to million dollars,
z_zones = df_dispatch_zones.iloc[:, [i for i in range(1, 12)]].to_numpy()

# Names of the input variables
input_labels = [
    'pmax','pmin_multi','ramp_multi','min_up_time','min_down_multi',
    'marg_cst','no_load_cst','startup_cst',
]

# List of basis functions for ALAMO
alamo_config = {
    "constant": True,
    "linfcns": True,
    "multi2power": (1, 2, 3),
    "monomialpower": (2, 3),
    "maxterms": 15,
}

# =====================================================================
#      TRAIN A SURROGATE FOR THE NUMBER OF STARTUPS
# =====================================================================

# 1/3 of data used for test
X_train, X_test, z_train, z_test = train_test_split(x, z_nstartups, test_size=0.33, random_state=42)

# Train ALAMO surrogates
# NOTE: The input-output data will be shifted by mean and scaled by the standard deviation by default.
#       If this is not desired, set shift_scale_data argument to False
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    x_labels=input_labels,
    z_labels=['nstartups'],
    trainer="alamo",
    alamo_config=alamo_config,
    filename="nstartups",
)

# Train KERAS surrogate
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="keras",
    activation="tanh",
    hidden_layers=(100, 50),
    filename="nstartups",
)

# Train SCIKIT surrogate
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="scikit",
    activation="tanh",
    hidden_layers=(100, 50),
    filename="nstartups",
)

# =====================================================================
#      TRAIN A SURROGATE FOR THE REVENUE
# =====================================================================

# 1/3 of data used for test
X_train, X_test, z_train, z_test = train_test_split(x, z_revenue, test_size=0.33, random_state=42)

# Train ALAMO surrogates
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    x_labels=input_labels,
    z_labels=['revenue'],
    trainer="alamo",
    alamo_config=alamo_config,
    filename="revenue",
)

# Train KERAS surrogate
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="keras",
    activation="tanh",
    hidden_layers=(100, 50),
    filename="revenue",
)

# Train SCIKIT surrogate
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="scikit",
    activation="tanh",
    hidden_layers=(100, 50),
    filename="revenue",
)

# =====================================================================
#      TRAIN A SURROGATE FOR EACH ZONE
# =====================================================================

# 1/3 of data used for test
X_train, X_test, z_train, z_test = train_test_split(x, z_zones, test_size=0.33, random_state=42)
alamo_config["maxterms"] = [15] * 11

# Train ALAMO surrogates
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    x_labels=input_labels,
    z_labels=['zone_{}'.format(zone) for zone in range(11)],
    trainer="alamo",
    alamo_config=alamo_config,
    filename="zones",
)

# Train KERAS surrogate
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="keras",
    activation="tanh",
    hidden_layers=(100, 100),
    filename="zones",
)

# Train SCIKIT surrogate
build_surrogate_model(
    x_data=X_train,
    z_data=z_train,
    trainer="scikit",
    activation="tanh",
    hidden_layers=(100, 100),
    filename="zones",
)
