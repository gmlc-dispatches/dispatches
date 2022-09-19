import multiprocessing
from parameters import generator_name, pem_name, pem_name_from_grid index_mapper

from dispatches.prescient_sweeps.utils import summarize_results

base_directories = [
        "parameter_sweep_10_500",
        "parameter_sweep_10_1000",
        "parameter_sweep_15_500",
        "parameter_sweep_15_1000",
]

bus_name = "Caesar"

for base_dir in base_directories:
    output_dir = "results_"+base_dir
    multiprocessing.Process(target=summarize_results, args=(base_dir, index_mapper, generator_name, bus_name, output_dir, pem_name, pem_name_from_grid)).start()
