import os
import sys

from parameters import update_function
from dispatches.prescient_sweeps.prescient_options_10_reserves_500_shortfall import prescient_options
from dispatches.prescient_sweeps.utils import run_sweep

bad_guys = [21, 129, 141, 163, 237, 282, 366, 415, 518, 531, 558, 630, 752, 753, 869, 1000, 1065, 1078, 1091, 1104, 1117, 1130, 1143, 1156, 1169] 

if __name__ == "__main__":
    prescient_options["output_directory"] = str(os.path.splitext(os.path.basename(__file__))[0])
    
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    if stop > len(bad_guys):
        stop = len(bad_guys)    
    
    run_sweep(update_function, prescient_options, bad_guys[start:stop])
