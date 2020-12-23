"""
Wrappers for Prescient RTS-GMLC functions
"""
# stdlib
import os
from pathlib import Path
from typing import Dict
# third-party
import prescient.downloaders.rts_gmlc as rts_downloader
from prescient.downloaders.rts_gmlc_prescient.rtsgmlc_to_dat import write_template


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


# Actions
# -------

class Action:
    """Base class
    """
    def __init__(self, ds_list):
        self._ds_list = ds_list

    def run(self) -> Dict:
        """

        Returns:
            Dict where keys are names of output datasets, and values are pairs (directory, [file names..]),
            where files and directories are Path objects.
        """

        pass


class CreateTemplate(Action):
    def __init__(self, ds_list):
        assert len(ds_list) == 1
        super().__init__(ds_list)

    def run(self, **kwargs):
        ds = self._ds_list[0]
        directory = ds.meta["directory"]
        target = Path(directory) / "_templates" / "rts_with_network_template_hotstart.dat"
        write_template(rts_gmlc_dir=directory, file_name=str(target))
        return {"dat_file": (target.parent, [target.name])}


class Actions:
    create_template = CreateTemplate
    create_time_series = None
    copy_scripts = None