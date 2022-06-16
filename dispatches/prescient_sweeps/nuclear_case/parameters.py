from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

generator_name = "121_NUCLEAR_1"

sweep_params = {
        "PEM_fraction" : [ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50 ],
        "PEM_indifference_point" : [ 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 ],
        }

index_mapper = FlattenedIndexMapper(sweep_params)

def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]
    point = index_mapper(index)

    assert "p_min" in gen
    PEM_capacity = point["PEM_fraction"] * gen["p_max"]
    gen["p_min"] = gen["p_max"] - PEM_capacity

    if "p_cost" not in gen:
        del gen["p_fuel"]
        gen["p_cost"] = { "data_type" : "cost_curve", "cost_curve_type" : "piecewise" }
    gen["p_cost"]["values"] = [[gen["p_min"], 0.], [gen["p_max"], PEM_capacity*point["PEM_indifference_point"]]]

    gen["ramp_up_60min"] = PEM_capacity*60
    gen["ramp_down_60min"] = PEM_capacity*60
    gen["fixed_commitment"] = {"data_type" : "time_series", "values" : [1]*len(model_data.data["system"]["time_keys"])}
