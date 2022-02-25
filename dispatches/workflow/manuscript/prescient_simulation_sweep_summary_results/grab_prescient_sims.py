import shutil
import os

pmax_low = [[52216, 52936, 53656, 54376, 55096, 55816],
 [39256, 39976, 40696, 41416, 42136, 42856],
 [26296, 27016, 27736, 28456, 29176, 29896],
 [13336, 14056, 14776, 15496, 16216, 16936],
 [376, 1096, 1816, 2536, 3256, 3976]]

pmax_high = [[52219, 52939, 53659, 54379, 55099, 55819],
 [39259, 39979, 40699, 41419, 42139, 42859],
 [26299, 27019, 27739, 28459, 29179, 29899],
 [13339, 14059, 14779, 15499, 16219, 16939],
 [379, 1099, 1819, 2539, 3259, 3979]]

for start_cst in pmax_low:
    for i in start_cst:
        src = "../../prescient_sweeps_raw_output/deterministic_simulation_output_index_{}".format(i)
        dst = os.path.basename(src)
        shutil.copytree(src, dst)

for start_cst in pmax_high:
    for i in start_cst:
        src = "../../prescient_sweeps_raw_output/deterministic_simulation_output_index_{}".format(i)
        dst = os.path.basename(src)
        shutil.copytree(src, dst)
