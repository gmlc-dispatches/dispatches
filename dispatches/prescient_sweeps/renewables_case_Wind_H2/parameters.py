import copy
from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

generator_name = "303_WIND_1"
pem_name = generator_name + "_PEM"
installed_capacity = 847.0

sweep_params = {
        "PEM_power_capacity" : [val*installed_capacity for val in 
                                    [ 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 ]],
        "PEM_bid" : [ 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0 ],
        }

index_mapper = FlattenedIndexMapper(sweep_params)


def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]
    point = index_mapper(index)

    pem = copy.deepcopy(gen)

    pem["p_min"] = 0.0
    pem["p_max"]["values"] = [ min(point["PEM_power_capacity"], val) for val in gen["p_max"]["values"] ]
    pem["p_cost"] = point["PEM_bid"]

    model_data.data["elements"]["generator"][pem_name] = pem

    for idx, val in enumerate(gen["p_max"]["values"]):
        gen["p_max"]["values"][idx] = max(0., val - point["PEM_power_capacity"])
