import copy
from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

generator_name = "319_PV_1"
battery_name = generator_name + "_Battery"
installed_capacity = 188.2

sweep_params = {
        "battery_power_capacity" : [val*installed_capacity for val in 
                                    [ 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 ]],
        "battery_discharge_bid" : [ 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0 ],
        "battery_bid_spread" : [ 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0 ],
        }

index_mapper = FlattenedIndexMapper(sweep_params)


def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]
    point = index_mapper(index)

    battery = copy.deepcopy(gen)

    battery["p_min"] = 0.0
    battery["p_max"]["values"] = [ point["battery_power_capacity"] for _ in battery["p_max"]["values"] ]
    battery["p_cost"] = point["battery_discharge_bid"]
    battery["generator_type"] = "virtual"

    battery_charge = copy.deepcopy(battery)

    battery_charge["p_max"] = 0.0
    battery_charge["p_cost"] = point["battery_discharge_bid"] - point["battery_bid_spread"]

    model_data.data["elements"]["generator"][battery_name+"_discharge"] = battery
    model_data.data["elements"]["generator"][battery_name+"_charge"] = battery_charge

    # get time-varying data structure
    battery_charge["p_min"] = copy.deepcopy(gen["p_max"])
    pv_avail = gen["p_max"]["values"][0]
    for idx, val in enumerate(gen["p_max"]["values"]):
        excess_PV = max(0., val - point["battery_power_capacity"])
        gen["p_max"]["values"][idx] = excess_PV
        PV_charging = min(point["battery_power_capacity"], val)
        battery_charge["p_min"]["values"][idx] = -(point["battery_power_capacity"] - PV_charging) 
    print(f"pv: {pv_avail}, new pv bid: {gen['p_max']['values'][0]}, battery_bid: {battery['p_max']['values'][0]}, battery_charge bid: {battery_charge['p_min']['values'][0]}")
