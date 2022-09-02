import multiprocessing
from parameters import generator_name, battery_name, index_mapper

from dispatches.prescient_sweeps.utils import summarize_results

base_directories = [
        "parameter_sweep_10_500",
        "parameter_sweep_10_1000",
        "parameter_sweep_15_500",
        "parameter_sweep_15_1000",
]

bus_name = "Clay"

for base_dir in base_directories:
    output_dir = "results_"+base_dir
    multiprocessing.Process(target=summarize_results, args=(base_dir, index_mapper, generator_name, bus_name, output_dir, battery_name)).start()
