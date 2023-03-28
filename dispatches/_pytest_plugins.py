#################################################################################
# DISPATCHES was produced under the DOE Design Integration and Synthesis Platform
# to Advance Tightly Coupled Hybrid Energy Systems program (DISPATCHES), and is
# copyright (c) 2020-2023 by the software owners: The Regents of the University
# of California, through Lawrence Berkeley National Laboratory, National
# Technology & Engineering Solutions of Sandia, LLC, Alliance for Sustainable
# Energy, LLC, Battelle Energy Alliance, LLC, University of Notre Dame du Lac, et
# al. All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. Both files are also available online at the URL:
# "https://github.com/gmlc-dispatches/dispatches".
#################################################################################
from typing import Iterable

import pytest


class Matcher:
    def __init__(self, specs: Iterable):
        self._specs = specs

    def __iter__(self):
        return iter(self._specs)

    def __call__(self, item: pytest.Item) -> bool:
        raise NotImplementedError()


class Keywords(Matcher):
    def __call__(self, item: pytest.Item):
        kws = item.keywords
        for spec in self:
            to_match = [spec] if isinstance(spec, str) else list(spec)
            if all(
                kw in kws
                for kw in to_match
            ):
                return True
        return False


class Issue178:
    matcher = Keywords([
        "design_ultra_supercritical_power_plant.ipynb",
        ("test_charge_usc_powerplant.py", "test_main_function"),
        ("test_charge_usc_powerplant.py", "test_initialize"),
        ("test_charge_usc_powerplant.py", "test_costing"),
        ("test_charge_usc_powerplant.py", "test_usc_charge_model"),
        ("test_discharge_usc_powerplant.py", "test_main_function"),
        ("test_discharge_usc_powerplant.py", "test_initialize"),
        ("test_discharge_usc_powerplant.py", "test_costing"),
        ("test_discharge_usc_powerplant.py", "test_usc_discharge_model"),
    ])
    marker = pytest.mark.xfail(run=False, reason="known issue #178")

    def __init__(self):
        self.matched = []

    def pytest_addoption(self, parser: pytest.Parser):
        parser.addoption(
            "--xfail-issue-178",
            help="Apply xfail marker to items affected by known issue 178",
            action="store_true",
            default=False,
            dest="xfail_issue_178",
        )

    def pytest_configure(self, config: pytest.Config):
        if not config.option.xfail_issue_178:
            config.pluginmanager.unregister(self)

    def pytest_collection_modifyitems(self, config: pytest.Config, items: Iterable[pytest.Item]):
        self.matched = [item for item in items if self.matcher(item)]
        for item in self.matched:
            item.add_marker(self.marker)

    def pytest_report_collectionfinish(self, config: pytest.Config):
        lines = [
            "The following markers were applied:",
        ]
        lines += [f"{self.marker.mark}: {len(self.matched)} items"]
        return lines


issue_178 = Issue178()
