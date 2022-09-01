import copy
from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

generator_name = "319_PV_1"
battery_name = generator_name + "_Battery"
installed_capacity = 188.2

sweep_params = {
        "battery_power_capacity" : [val*installed_capacity for val in 
                                    [ 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 ]],
        "battery_bid" : [ 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0 ],
        }

index_mapper = FlattenedIndexMapper(sweep_params)


def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]
    point = index_mapper(index)

    battery = copy.deepcopy(gen)

    battery["p_min"] = 0.0
    battery["p_max"]["values"] = [ point["battery_power_capacity"] for _ in battery["p_max"]["values"] ]
    battery["p_cost"] = point["battery_bid"]

    model_data.data["elements"]["generator"][battery_name] = battery

    for idx, val in enumerate(gen["p_max"]["values"]):
        gen["p_max"]["values"][idx] = max(0., val - point["battery_power_capacity"])
