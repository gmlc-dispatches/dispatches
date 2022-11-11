import glob
import os

from parameters import index_mapper

def get_failed_indices(base_dir):
    all_indices = set(range(len(index_mapper)))
    found_indices = set()
    for i in all_indices:
        fn = f"{base_dir}_index_{i}/overall_simulation_output.csv"
        if os.path.isfile(fn):
            found_indices.add(i)

    return all_indices - found_indices

if __name__ == "__main__":
    import sys
    bad_indices = get_failed_indices(sys.argv[1])
    print(f"bad indices: {sorted(list(bad_indices))}")
