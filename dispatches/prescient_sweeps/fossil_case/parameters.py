import copy
from dispatches.prescient_sweeps.utils import FlattenedIndexMapper

generator_name = "223_STEAM_3"
discharge_unit_name = "223_STEAM_2"
non_generator_name = "223_STEAM_1"

sweep_params = {
        "Storage_size" : [ i*20.0 for i in range(1, 11) ],
        "Discharge_adder" : [ 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0 ], 
        "Charge_adder" : [ -15.0, -10.0, -5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0 ],
        }

base_generator_p_max = 436
base_generator_p_min = 284

base_generator_op_cost_at_p_max = 9053.830223 
base_generator_op_cost_at_p_min = 6236.99717

base_generator_marginal_cost = ((base_generator_op_cost_at_p_max - base_generator_op_cost_at_p_min)
        / (base_generator_p_max - base_generator_p_min))

index_mapper = FlattenedIndexMapper(sweep_params)

def update_function(model_data, index):
    gen = model_data.data["elements"]["generator"][generator_name]

    point = index_mapper(index)
    storage_size = point["Storage_size"]

    gen["p_min"] = base_generator_p_min - storage_size
    gen["p_max"] = base_generator_p_max
    gen["p_min_agc"] = gen["p_min"]
    gen["p_max_agc"] = gen["p_max"]
    gen["agc_capable"] = False
    gen["startup_capacity"] = base_generator_p_min + gen["ramp_up_60min"]*0.5
    gen["shutdown_capacity"] = base_generator_p_max*2

    # ramp rate exceeds p_min - p_max, so just ignore
    # (no ramp rate on switching to charging from grid)
    gen["ramp_up_60min"] = gen["p_max"] - gen["p_min"]
    gen["ramp_down_60min"] = gen["p_max"] - gen["p_min"]

    if "p_cost" not in gen:
        del gen["p_fuel"]
        gen["p_cost"] = { "data_type" : "cost_curve", "cost_curve_type" : "piecewise" }

    charge_marginal_cost = base_generator_marginal_cost + point["Charge_adder"]
    if charge_marginal_cost > base_generator_marginal_cost:
        gen["p_cost"]["values"] = [ 
            [
                base_generator_p_min - storage_size,
                (base_generator_op_cost_at_p_max 
                    - charge_marginal_cost*storage_size
                    - base_generator_marginal_cost*(base_generator_p_max - base_generator_p_min))
            ],
            [
                base_generator_p_max - storage_size,
                (base_generator_op_cost_at_p_max
                    - charge_marginal_cost*(storage_size))
            ], 
            [
                base_generator_p_max,
                base_generator_op_cost_at_p_max
            ],
        ]
    elif charge_marginal_cost < base_generator_marginal_cost:
        gen["p_cost"]["values"] = [ 
            [
                base_generator_p_min - storage_size,
                (base_generator_op_cost_at_p_max 
                    - base_generator_marginal_cost*(base_generator_p_max-base_generator_p_min)
                    - charge_marginal_cost*storage_size)
            ], 
            [
                base_generator_p_min,
                (base_generator_op_cost_at_p_max
                    - base_generator_marginal_cost*(base_generator_p_max - base_generator_p_min))
            ], 
            [
                base_generator_p_max,
                base_generator_op_cost_at_p_max
            ],
        ]

    else: #Charge_adder == 0
        gen["p_cost"]["values"] = [ 
            [
                base_generator_p_min - storage_size,
                (base_generator_op_cost_at_p_max
                    - base_generator_marginal_cost*(base_generator_p_max - base_generator_p_min - storage_size))
            ], 
            [
                base_generator_p_max,
                base_generator_op_cost_at_p_max
            ],
        ]

    discharge_gen = model_data.data["elements"]["generator"][discharge_unit_name]
    discharge_marginal_cost = base_generator_marginal_cost + max(0., point["Charge_adder"]) + point["Discharge_adder"]

    if "startup_fuel" in discharge_gen:
        del discharge_gen["startup_fuel"]
    if "p_cost" not in discharge_gen:
        del discharge_gen["p_fuel"]
        discharge_gen["p_cost"] = { "data_type" : "cost_curve", "cost_curve_type" : "piecewise" }
    discharge_gen["startup_cost"] = 0.0
    discharge_gen["minimum_up_time"] = 0
    discharge_gen["minimum_down_time"] = 0
    discharge_gen["p_min"] = 0.0
    discharge_gen["p_max"] = storage_size
    discharge_gen["p_min_agc"] = 0.0
    discharge_gen["p_max_agc"] = storage_size
    discharge_gen["agc_capable"] = False
    discharge_gen["initial_p_output"] = min(storage_size, discharge_gen["initial_p_output"])
    discharge_gen["startup_capacity"] = storage_size
    discharge_gen["shutdown_capacity"] = storage_size
    discharge_gen["ramp_up_60min"] = storage_size*60
    discharge_gen["ramp_down_60min"] = storage_size*60

    discharge_gen["fixed_commitment"] = {"data_type" : "time_series", "values" : [1]*len(model_data.data["system"]["time_keys"])}
    discharge_gen["p_cost"]["values"] = [[0.0, 0.0], [storage_size, discharge_marginal_cost*storage_size]]


    non_gen = model_data.data["elements"]["generator"][non_generator_name]
    non_gen["fixed_commitment"] = {"data_type" : "time_series", "values" : [0]*len(model_data.data["system"]["time_keys"])}
    non_gen["p_min"] = 0.0
    non_gen["p_max"] = 0.0
    non_gen["p_min_agc"] = 0.0
    non_gen["p_max_agc"] = 0.0
    non_gen["initial_p_output"] = 0.0
    non_gen["startup_capacity"] = 0.0 
    non_gen["shutdown_capacity"] = 0.0 
    non_gen["ramp_up_60min"] = 0.0
    non_gen["ramp_down_60min"] = 0.0
    non_gen["p_fuel"]["values"] = [[0.0, 0.0]]

    #print(point)
    #print(gen)
    #print(discharge_gen)
    #print(non_gen)
