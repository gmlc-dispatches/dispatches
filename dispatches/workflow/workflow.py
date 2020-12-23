"""
Managed data workflows for Prescient
"""
# stdlib
import os
from pathlib import Path
# third-party
from idaes.dmf import DMF, resource
# package
from . import rts_gmlc


class DatasetType:
    RTS_GMLC = "rts-gmlc"


class ManagedWorkflow:
    def __init__(self, name, workspace_path):
        self._name = name
        p = Path(workspace_path)
        self._dmf = DMF(p, create=True)
        self._workspace_name = p.name

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
        existing = self._dmf.find_one(name=type_)
        if existing:
            print("Already have RTS-GMLC resource")
            return Dataset.from_resource(existing)
        dsf = DatasetFactory(type_, workflow=self)
        ds = dsf.create(target_path=self._download_path(), **kwargs)
        self._add_to_dmf(ds)
        return ds

    def _download_path(self):
        return self._dmf.workspace_path / "downloads"

    def _add_to_dmf(self, ds):
        datafile_list = [{
                "path": filename,
                "desc": f"{DatasetType.RTS_GMLC} file {filename}"}
                for filename in ds.meta["files"]]
        # print(f"DATAFILES: {datafile_list}")
        r = resource.Resource({
            "datafiles": datafile_list,
            "datafiles_dir": str(Path(ds.meta["directory"]).resolve()),
        })
        r.set_field("name", ds.name)
        self._dmf.add(r)


class Dataset:
    def __init__(self, name):
        self.name = name
        self._meta = {}

    @property
    def meta(self):
        return self._meta.copy()

    def add_meta(self, key, value):
        self._meta[key] = value

    @staticmethod
    def from_resource(r):
        ds = Dataset(name=r.name)
        ds.add_meta("directory", r.v["datafiles_dir"])
        ds.add_meta("files", [f["path"] for f in r.v["datafiles"]])
        return ds

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
        if name == DatasetType.RTS_GMLC:

            def download_fn(target_path=None, **kwargs):
                rts_gmlc_dir = rts_gmlc.download(target_path)
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
