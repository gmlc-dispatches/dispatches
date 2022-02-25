from prescient.simulator import Prescient

# NOTE: `rts_gmlc_data_dir` should point to a directory containing RTS-GMLC scenarios
rts_gmlc_data_dir = "/home/xgao1/DowlingLab/RTS-GMLC/RTS_Data/SourceData"

options = {
    "data_path": rts_gmlc_data_dir,
    "input_format":"rts-gmlc",
    "simulate_out_of_sample": True,
    "run_sced_with_persistent_forecast_errors": True,
    "output_directory": "bidding_plugin_test_multiperiod_wind_battery",
    "start_date": "01-02-2020",
    "num_days": 7,
    "sced_horizon": 4,
    "ruc_horizon": 48,
    "compute_market_settlements": True,
    "day_ahead_pricing": "LMP",
    "ruc_mipgap": 0.05,
    "symbolic_solver_labels": True,
    "reserve_factor": 0.0,
    "deterministic_ruc_solver": "gurobi",
    "sced_solver": "gurobi",
    "plugin": {
        "doubleloop": {
            "module": "plugin_wind_battery_doubleloop.py",
            "bidding_generator": "309_WIND_1",
        }
    },
}

Prescient().simulate(**options)
