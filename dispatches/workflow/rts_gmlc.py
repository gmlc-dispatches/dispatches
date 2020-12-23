"""
Wrappers for Prescient RTS-GMLC functions
"""
import os
from pathlib import Path
import prescient.downloaders.rts_gmlc as rts_downloader


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
