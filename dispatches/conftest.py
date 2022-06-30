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
from _pytest.config import Config


_MARKERS = {
    'unit': 'quick tests that do not require a solver, must run in < 2 s',
    'component': 'quick tests that may require a solver',
    'integration': 'long duration tests',
}


def pytest_configure(config: Config):

    for name, descr in _MARKERS.items():
        config.addinivalue_line(
            'markers', f'{name}: {descr}'
        )
