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
Managed data workflows for Prescient
"""
# stdlib
import os
# package
from . import rts_gmlc


class ManagedWorkflow:
    def __init__(self, name, workspace_name):
        self._name = name
        self._workspace_name = workspace_name
        self._datasets = {}
        # TODO: create instance of DMF

    @property
    def name(self):
        return self._name

    @property
    def workspace_name(self):
        return self._workspace_name

    def get_dataset(self, type_, **kwargs):
        """Creates and returns a dataset of the specified type. If called more than once with the
        same type of dataset, then returns the previous value.
        """
        ds = self._datasets.get(type_, None)
        if ds is not None:
            return ds
        dsf = DatasetFactory(type_, workflow=self)
        ds = dsf.create(**kwargs)
        self._datasets[type_] = ds
        # TODO: register new dataset with DMF
        return ds


class Dataset:
    def __init__(self, name):
        self.name = name
        self._meta = {}

    @property
    def meta(self):
        return self._meta.copy()

    def add_meta(self, key, value):
        self._meta[key] = value

    def __str__(self):
        lines = [
            "Metadata",
            "--------"
        ]
        for key, value in self._meta.items():
            lines.append("%s:" % key)
            lines.append(str(value))
        return "\n".join(lines)


class DatasetFactory:
    def __init__(self, type_, workflow=None):
        self._wf = workflow
        try:
            self.create = self._get_factory_function(type_)
        except KeyError:
            raise KeyError("Cannot create dataset of type '%s'" % type_)

    @classmethod
    def _get_factory_function(cls, name):
        # This could be more dynamic..
        if name == "rts-gmlc":

            def download_fn(**kwargs):
                rts_gmlc_dir = rts_gmlc.download()
                dataset = Dataset(name)
                dataset.add_meta("directory", rts_gmlc_dir)
                dataset.add_meta("files", os.listdir(rts_gmlc_dir))
                return dataset

            return download_fn
        elif name == "null":
            def fn(**kwargs):
                return None  # XXX: or do we need a NullDataset subclass?
            return fn
        else:
            raise KeyError(name)
