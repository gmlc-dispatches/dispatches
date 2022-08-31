#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis
# Platform to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES),
# and is copyright (c) 2022 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#
#################################################################################
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
