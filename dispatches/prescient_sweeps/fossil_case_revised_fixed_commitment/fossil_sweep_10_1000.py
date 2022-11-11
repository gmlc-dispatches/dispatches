import os
import sys

from parameters import update_function
from dispatches.prescient_sweeps.prescient_options_10_reserves_1000_shortfall import prescient_options
from dispatches.prescient_sweeps.utils import run_sweep

bad_guys = [142, 186, 202, 210, 244, 324, 348, 428, 464, 561, 568, 634, 694, 778, 869, 906, 1065, 1078, 1091, 1104, 1117, 1130, 1143, 1156, 1169] 

if __name__ == "__main__":
    prescient_options["output_directory"] = str(os.path.splitext(os.path.basename(__file__))[0])
    
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    if stop > len(bad_guys):
        stop = len(bad_guys)    
    
    run_sweep(update_function, prescient_options, bad_guys[start:stop])
