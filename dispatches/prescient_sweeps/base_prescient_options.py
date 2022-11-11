import os
from dispatches.prescient_sweeps.download_rts_gmlc import download_rts_gmlc
download_rts_gmlc()

this_file_path = os.path.dirname(os.path.realpath(__file__))
# default some options
prescient_options = {
        "data_path":os.path.join(this_file_path, "DISPATCHES_RTS-GMLC/RTS_Data/SourceData"),
        "reserve_factor":None,
        "simulate_out_of_sample":True,
        "output_directory":None,
        "monitor_all_contingencies":False,
        "input_format":"rts-gmlc",
        "start_date":"01-01-2020",
        "num_days":366,
        "sced_horizon":1,
        "ruc_mipgap":0.01,
	    "deterministic_ruc_solver": "xpress_persistent",
	    "deterministic_ruc_solver_options" : {"threads":2, "heurstrategy":2, "cutstrategy":3, "symmetry":2, "maxnode":1000},
        "sced_solver":"xpress_persistent",
        "sced_frequency_minutes":60,
	    "sced_solver_options" : {"threads":1},
        "ruc_horizon":36,
        "compute_market_settlements":True,
        "output_solver_logs":False,
        "price_threshold":None,
        "transmission_price_threshold":None,
        "contingency_price_threshold":None,
        "reserve_price_threshold":None,
        "day_ahead_pricing":"aCHP",
        "enforce_sced_shutdown_ramprate":False,
        "ruc_slack_type":"ref-bus-and-branches",
        "sced_slack_type":"ref-bus-and-branches",
	    "disable_stackgraphs":True,
        }
