from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

generator_name = "121_NUCLEAR_1"

sweep_params = {
        "PEM_fraction" : [ 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50 ],
        "PEM_indifference_point" : [ 15.0, 20.0, 25.0, 30.0, 35.0, 40.0 ],
        "Fuel_cell_efficiency" : [ 0.25, 0.35, 0.45],
        "Fuel_cell_fraction" : [ 0.25, 0.50, 0.75, 1.0 ],
        }

index_mapper = FlattenedIndexMapper(sweep_params)

def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]
    point = index_mapper(index)

    gen_pmax = gen["p_max"]

    PEM_capacity = point["PEM_fraction"] * gen_pmax
    Fuel_cell_capacity = point["Fuel_cell_fraction"] * PEM_capacity 
    fuel_cell_marginal_cost = point["PEM_indifference_point"] / point["Fuel_cell_efficiency"]

    assert "p_min" in gen
    gen["p_min"] = gen_pmax - PEM_capacity
    gen["p_max"] = gen_pmax + Fuel_cell_capacity

    cost_at_pmin = 0.0
    cost_at_gen_pmax = cost_at_pmin + PEM_capacity*point["PEM_indifference_point"]
    cost_at_pmax = cost_at_gen_pmax + Fuel_cell_capacity*fuel_cell_marginal_cost

    if "p_cost" not in gen:
        del gen["p_fuel"]
        gen["p_cost"] = { "data_type" : "cost_curve", "cost_curve_type" : "piecewise" }
    gen["p_cost"]["values"] = [[gen["p_min"], cost_at_pmin],
                               [gen_pmax, cost_at_gen_pmax], 
                               [gen["p_max"], cost_at_pmax],
                              ]

    gen["ramp_up_60min"] = (PEM_capacity+Fuel_cell_capacity)*60
    gen["ramp_down_60min"] = (PEM_capacity+Fuel_cell_capacity)*60
    gen["fixed_commitment"] = {"data_type" : "time_series", "values" : [1]*len(model_data.data["system"]["time_keys"])}


if __name__ == "__main__":
    for idx, val in index_mapper.all_points_generator():
        print( idx, val )
