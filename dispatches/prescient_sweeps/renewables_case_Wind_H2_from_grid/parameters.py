import copy
from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

import math

generator_name = "303_WIND_1"
pem_name = generator_name + "_Wind_to_PEM"
pem_name_from_grid = generator_name + "_grid_to_PEM"
installed_capacity = 847.0

sweep_params = {
        "PEM_power_capacity" : [val*installed_capacity for val in 
                                    [ 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 ]],
        "PEM_bid_from_wind" : [ 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0 ],
        "PEM_bid_from_grid_spread" : [ 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0 ],
        }

index_mapper = FlattenedIndexMapper(sweep_params)


def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]
    point = index_mapper(index)
    print(point)

    pem = copy.deepcopy(gen)


    pem["p_min"] = { 
            "data_type" : "time_series",
            "values" : [ -min(point["PEM_power_capacity"], val) for val in gen["p_max"]["values"] ],
            }
    pem["p_max"] = 0.0
    pem["p_cost"] = point["PEM_bid_from_wind"]
    pem["generator_type"] = "virtual"
    pem["fuel"] = "Other"

    model_data.data["elements"]["generator"][pem_name] = pem

    pem_from_grid = copy.deepcopy(pem)

    pem_from_grid["p_min"] = { 
            "data_type" : "time_series",
            "values" : [ -(val + point["PEM_power_capacity"]) for val in pem["p_min"]["values"] ],
            }
    pem_from_grid["p_max"] = 0.0
    pem_from_grid["p_cost"] = point["PEM_bid_from_wind"] - point["PEM_bid_from_grid_spread"]
    pem_from_grid["fuel"] = "Other"

    model_data.data["elements"]["generator"][pem_name_from_grid] = pem_from_grid

    print(pem_name, pem)
    print(pem_name_from_grid, pem_from_grid)

    assert all(math.isclose(pfw+pfg, -point["PEM_power_capacity"], abs_tol=1e-3) for pfw, pfg in zip(pem["p_min"]["values"], pem_from_grid["p_min"]["values"]))
