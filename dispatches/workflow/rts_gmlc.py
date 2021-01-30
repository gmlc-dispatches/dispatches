"""
Wrappers for Prescient RTS-GMLC functions
"""
# stdlib
import os
from pathlib import Path
from subprocess import Popen, PIPE
import sys
from typing import Dict
# third-party
import prescient.downloaders.rts_gmlc as rts_downloader
from prescient.downloaders.rts_gmlc_prescient.rtsgmlc_to_dat import write_template
from prescient.downloaders.rts_gmlc_prescient.process_RTS_GMLC_data import create_timeseries
from prescient.scripts.runner import parse_commands


def download(target_path) -> Path:
    """Wraps RTS GMLC downloader.

    Args:
        target_path: Where downloads go.

    Returns:
        Path to root of RTS-GMLC download.
    """
    if target_path is None:
        target_path = Path(os.getcwd())
    else:
        target_path = Path(target_path)  # convert str to Path
    rts_downloader.rts_download_path = str(target_path.absolute())
    rts_downloader.download()
    rts_gmlc_dir = Path(rts_downloader.rts_download_path) / "RTS-GMLC"
    return rts_gmlc_dir


# Processing functions
# --------------------

def download_path():
    return Path(rts_downloader.rts_download_path)


# Note: 'ds' parameter is not used, but it is an anchor for relationships in
# the workflow, so passed in to these functions anyways.

def create_template(ds):
    directory = download_path() / "templates"
    target = directory / "rts_with_network_template_hotstart.dat"
    source = download_path() / "RTS-GMLC"
    write_template(rts_gmlc_dir=str(source), file_name=str(target))
    return {"dat_file": (target.parent, [target.name])}


def create_time_series(ds):
    create_timeseries(download_path())
    output_path = download_path() / "timeseries_data_files"
    output_files = output_path.glob("*")
    return {"output_files": (output_path, output_files)}


def copy_scripts(ds):
    rts_downloader.copy_templates()
    output_path = download_path()
    output_files = output_path.glob("*")
    return {"output_files": (output_path, output_files)}


def runner(datasets, **kwargs):
    """Run script on a list of datasets.
    """
    for ds in datasets:
        config = ds.meta["files"][0]  # TODO: error checking??
        config_dir = ds.meta["directory"]
        config_path = config_dir / config
        if not config_path.exists() or not config_path.is_file():
            raise ValueError(f"Configuration file '{config_path}' is not a file or does not exist")
        _run_script(config_path, **kwargs)


def _run_script(path: Path, collector=None, **kwargs):
    """Based on behavior of Prescient's prescient.scripts.runner
    """
    script, options = parse_commands(path)
    # Assume 'script' is in our execution PATH? Append .exe to its name for Windows
    if sys.platform.startswith('win'):
        if script.endswith(".py"):
            script = script[:-3]
        script = script + ".exe"
    # os.environ['PYTHONUNBUFFERED'] = '1'
    # Run script, from download directory
    orig_dir = os.curdir
    os.chdir(download_path())
    proc = Popen([script] + options, stdout=PIPE, stderr=PIPE, **kwargs)
    if collector:
        collector.collect(proc)
    else:
        proc.wait()
    os.chdir(orig_dir)
