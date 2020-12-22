"""
Wrappers for Prescient RTS-GMLC functions
"""
from pathlib import Path
import prescient.downloaders.rts_gmlc as rts_downloader


def download() -> Path:
    """Wraps RTS GMLC downloader.
    """
    rts_downloader.download()
    rts_gmlc_dir = Path(rts_downloader.rts_download_path) / "RTS-GMLC"
    return rts_gmlc_dir
