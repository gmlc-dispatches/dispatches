import multiprocessing
from parameters import generator_name, discharge_unit_name, non_generator_name, index_mapper

from dispatches.prescient_sweeps.utils import summarize_results

base_directories = [
        "nuclear_sweep_10_500",
        "nuclear_sweep_10_1000",
        "nuclear_sweep_15_500",
        "nuclear_sweep_15_1000",
]

bus_name = "Bloch"

for base_dir in base_directories:
    output_dir = "results_"+base_dir
    multiprocessing.Process(target=summarize_results, args=(base_dir, index_mapper, generator_name, bus_name, output_dir, discharge_unit_name, non_generator_name)).start()
