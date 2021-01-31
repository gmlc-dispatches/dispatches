"""
DMF-wrapped Prescient workflow
"""
# stdlib
import logging
from pathlib import Path
import re
# deps
# import pandas as pd  - imported later
# pkg
from dispatches.workflow import ManagedWorkflow, OutputCollector, set_log_level
from dispatches.workflow import DatasetType as DT
from dispatches.workflow import rts_gmlc

set_log_level(logging.DEBUG, dest=2)


def run_workflow(workspace_name, output_file=None):
    # Create a new managed workflow in a DMF workspace with the given name
    workspace_path = Path(".") / workspace_name
    wf = ManagedWorkflow(
        "prescient_tutorial_workflow", workspace_path
    )
    # Fetch RTS-GMLC data
    ds = wf.get_dataset(DT.RTS_GMLC)
    # print("dataset:\n%s" % ds)
    # Print out the file-list, as in the original tutorial
    print("Files:")
    try:
        files = ds.meta["files"]
    except KeyError:
        print("No files")
    else:
        for filename in files:
            print(filename)
    # run, with anchor to previously created dataset
    wf.run(rts_gmlc.create_template, inputs=ds)
    wf.run(rts_gmlc.create_time_series, inputs=ds)
    wf.run(rts_gmlc.copy_scripts, inputs=ds)

    if output_file is None:
        collector = None
    else:
        collector = OutputCollector(phase_delim="CBC MILP Solver",
                                    stdout=output_file, stderr=output_file)
    # This creates the "input deck" for July 10, 2020 -- July 16, 2020 for the simulator
    # in the output directory `deterministic_with_network_scenarios`.
    populate_script = "populate_with_network_deterministic.txt"
    populate_dir = None
    with open(workspace_path / "downloads" / populate_script, "r") as f:
        for line in f:
            m = re.search(r"--output-directory (.*)", line)
            if m:
                populate_dir = m.group(1)
                break
    if populate_dir is None:
        print(f"ERROR: Cannot find output directory in '{populate_script}'")
        return
    wf.run_script(populate_script, collector=collector,
                  output_dirs=[(populate_dir, "*.dat"), (populate_dir, "*.csv")])
    # Run the simulator
    wf.run_script("simulate_with_network_deterministic.txt", collector=collector)


def analysis() -> bool:
    import pandas as pd
    # load in the output data for the lines
    line_detail_file = Path("deterministic_with_network_simulation_output") / "line_detail.csv"
    if not line_detail_file.exists():
        print(f"File '{line_detail_file}' is missing")
        return False
    line_flows = pd.read_csv(line_detail_file,
        index_col=[0, 1, 2],
    )

    # load in the source data for the lines
    line_attributes = pd.read_csv(
        Path("RTS-GMLC") / "RTS_Data" / "SourceData" / "branch.csv", index_col=0,
    )

    # get the line limits
    line_limits = line_attributes["Cont Rating"]

    # get a series of flows
    line_flows = line_flows["Flow"]
    print(line_flows)
    line_limits.index.name = "Line"
    print(line_limits)
    lines_relative_flow = line_flows / line_limits
    lines_near_limits_time = lines_relative_flow[
        (lines_relative_flow > 0.99) | (lines_relative_flow < -0.99)
    ]
    print(lines_near_limits_time)

    return True


if __name__ == "__main__":
    import argparse
    pr = argparse.ArgumentParser()
    pr.add_argument("directory", help="Workspace directory")
    pr.add_argument("-w", "--workflow", action="store_true", help="Run workflow")
    pr.add_argument("-a", "--analysis", action="store_true", help="Perform analysis")
    pr.add_argument("-o", "--output", help="Store script output in FILE", metavar="FILE",
                    default=None)
    args = pr.parse_args()
    if not args.workflow and not args.analysis:
        print("Nothing to do. Add -w/--workflow and/or -a/--analysis flags")
    if args.workflow:
        run_workflow(args.directory, output_file=args.output)
    if args.analysis:
        ok = analysis()
        if not ok:
            print("Analysis failed")
    print("\nDONE\n")
