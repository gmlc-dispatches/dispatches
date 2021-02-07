"""
DMF-wrapped Prescient workflow
"""
# stdlib
import logging
from pathlib import Path
from typing import Tuple
# deps
# import pandas as pd  - imported later
# pkg
from dispatches.workflow import ManagedWorkflow, OutputCollector, set_log_level
from dispatches.workflow import DatasetType as DT
from dispatches.workflow import rts_gmlc
from idaes.dmf.resource import Triple


def run_workflow(workspace_name, output_file=None) -> ManagedWorkflow:
    # Create a new managed workflow in a DMF workspace with the given name
    workspace_path = Path(".") / workspace_name
    wf = ManagedWorkflow(
        "prescient_tutorial_workflow", workspace_path,
        tag="dmf-prescient"
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
    pop_script = rts_gmlc.download_path() / "populate_with_network_deterministic.txt"
    try:
        pop_opt = rts_gmlc.extract_options(pop_script)
        pop_path = rts_gmlc.download_path() / pop_opt["output-directory"]
    except KeyError:
        print(f"ERROR: Cannot find 'output-directory' option in '{pop_script}'")
        return
    wf.run_script(pop_script, collector=collector,
                  output_dirs=[(pop_path, "*.dat"), (pop_path, "*.csv")])
    # Run the simulator
    wf.run_script("simulate_with_network_deterministic.txt", collector=collector)
    return wf


def analysis(output_file=None) -> bool:
    import pandas as pd

    root = rts_gmlc.download_path()
    # load in the output data for the lines
    line_detail_file = root / "deterministic_with_network_simulation_output" / "line_detail.csv"
    if not line_detail_file.exists():
        print(f"File '{line_detail_file}' is missing")
        return False
    line_flows = pd.read_csv(line_detail_file,
        index_col=[0, 1, 2],
    )

    # load in the source data for the lines
    line_attributes = pd.read_csv(
        root / "RTS-GMLC" / "RTS_Data" / "SourceData" / "branch.csv", index_col=0,
    )

    # get the line limits
    line_limits = line_attributes["Cont Rating"]

    # output a series of flows
    line_flows = line_flows["Flow"]

    if output_file:
        with open(output_file, "w") as f:
            line_flows.to_csv(f)
    else:
        print(line_flows.to_string())
        print("\nLine flows\n------------\n")
    line_limits.index.name = "Line"
    print("\nLine limits\n------------\n")
    print(line_limits.to_string())
    lines_relative_flow = line_flows / line_limits
    lines_near_limits_time = lines_relative_flow[
        (lines_relative_flow > 0.99) | (lines_relative_flow < -0.99)
    ]
    print("\nLines near limit\n-----------------\n")
    print(lines_near_limits_time.to_string())

    return True

def draw_graph(workflow, output_file:Path = None):
    from idaes.dmf.graph import Graphviz
    from idaes.dmf.resource import RR_OBJ, RR_ROLE, RR_SUBJ, RR_PRED, RR_ID

    def node_label(meta):
        if meta["aliases"]:
            label = meta["aliases"][0]
        elif meta["desc"]:
            label = meta["desc"][:32]
        else:
            label = meta["id_"][:8]
        return {"label": label}

    def edge_label(meta):
        return {"label": meta.get("relation", "")}

    relations = {}
    for resource in workflow.dmf.find({"tags": workflow.tags}):
        # add (triple, metadata) for each relation
        # avoiding duplicate triples, to relations
        for rel in resource.v["relations"]:
            is_subject = rel[RR_ROLE] == RR_SUBJ
            if is_subject:
                triple = Triple(resource.id, rel[RR_PRED], rel[RR_ID])
            else:
                triple = Triple(rel[RR_ID], rel[RR_PRED], resource.id)
            meta = {k: resource.v[k] for k in ("aliases", "desc", "id_")}
            meta["relation"] = triple.predicate
            relations[triple] = meta
    relations = [(k, v) for k, v in relations.items()]
    gv = Graphviz.from_related(relations, node_attr_fn=node_label, edge_attr_fn=edge_label)

    if output_file is None:
        print("-- DOT Graph begin --")
        print(str(gv))
        print("-- DOT Graph end --")
    else:
        with output_file.open("w") as f:
            f.write(str(gv))


if __name__ == "__main__":
    import argparse
    pr = argparse.ArgumentParser()
    pr.add_argument("directory", help="Workspace directory")
    pr.add_argument("-a", "--analysis", action="store_true", help="Perform analysis")
    pr.add_argument("-g", "--graph", action="store_true", help="Create graph of data stored in DMF")
    pr.add_argument("-G", "--graph_output", metavar="FILE", help="Store graph output in FILE")
    pr.add_argument("-o", "--output", help="Store script output in FILE", metavar="FILE",
                    default=None)
    pr.add_argument("-O", "--analysis-output", help="Store analysis output in FILE", metavar="FILE",
                    default=None)
    pr.add_argument("-v", dest="vb", action="count", default=0)
    pr.add_argument("-w", "--workflow", action="store_true", help="Run workflow")
    args = pr.parse_args()
    if args.vb > 1:
        set_log_level(logging.DEBUG, dest=2)
    elif args.vb > 0:
        set_log_level(logging.INFO, dest=2)
    else:
        set_log_level(logging.WARNING, dest=2)
    if not args.workflow and not args.analysis:
        print("Nothing to do. Add -w/--workflow and/or -a/--analysis flags")
    if args.workflow:
        workflow = run_workflow(args.directory, output_file=args.output)
        if args.graph:
            if args.graph_output:
                output_file = Path(args.graph_output)
            else:
                output_file = None
            draw_graph(workflow, output_file=output_file)
    if args.analysis:
        ok = analysis(output_file=args.analysis_output)
        if not ok:
            print("Analysis failed")
    print("\nDONE\n")
