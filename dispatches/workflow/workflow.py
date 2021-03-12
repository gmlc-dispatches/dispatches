"""
Managed data workflows for Prescient
"""
# stdlib
import logging
import os
from pathlib import Path
import re
import sys
import threading
from typing import List, Tuple, Union

# deps
from idaes.dmf import DMF, resource

# pkg
from . import rts_gmlc


pkg_name = "dispatches.workflow"
_log = logging.getLogger(pkg_name)


class DatasetType:
    RTS_GMLC = "rts-gmlc"


class Dataset:
    def __init__(self, name):
        self.name = name
        self._meta = {}

    @property
    def meta(self):
        return self._meta.copy()

    def add_meta(self, key, value):
        self._meta[key] = value

    @property
    def resource_id(self):
        if "resource" in self._meta:
            return self._meta["resource"].id
        return None

    @property
    def resource(self):
        return self._meta.get("resource", None)

    @resource.setter
    def resource(self, value):
        if "resource" in self._meta:
            raise ValueError("Cannot set 'resource' on a dataset more than once")
        self._meta["resource"] = value

    @staticmethod
    def from_resource(r):
        ds = Dataset(name=r.name)
        ds.add_meta("directory", r.v["datafiles_dir"])
        ds.add_meta("files", [f["path"] for f in r.v["datafiles"]])
        ds.resource = r
        return ds

    def __str__(self):
        lines = ["Metadata", "--------"]
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
                _log.debug(f"Creating null dataset with args: {kwargs}")
                dataset = Dataset("null")
                return dataset

            return fn
        elif name == "script":

            def fn(path: Path, **kwargs):
                _log.debug(f"Creating dataset for script '{path}'")
                dataset = Dataset(f"script:{path.name}")
                dataset.add_meta("directory", path.parent.resolve())
                dataset.add_meta("files", [path.name])
                return dataset

            return fn
        else:
            raise KeyError(name)


class OutputCollector:
    """Collect (or redirect) output from running a script
    """

    def __init__(self, phase_delim=None, stderr=True, stdout=True):
        if phase_delim:
            if hasattr(phase_delim, "lower"):
                self._delim = re.compile(phase_delim)
            elif hasattr(phase_delim, "search"):
                self._delim = phase_delim
            else:
                raise TypeError(
                    f"Argument for 'phase_delim' type={type(phase_delim)} "
                    f"value={phase_delim} should be a string or a "
                    f"regular expression"
                )
        # methods for dealing with stderr/stdout
        self._methods = {}
        for name, arg in (("err", stderr), ("out", stdout)):
            if arg is True:
                dest = getattr(sys, "std" + name)
                self._methods[name] = self._redirect, (dest, False)
            elif arg is False:
                self._methods[name] = self._vanish, ()
            else:  # assume a destination
                dest = open(arg, "w")
                self._methods[name] = self._redirect, (dest, sys.stdout)

    def collect(self, proc):
        """Collect output (stderr/stdout) from process.
        """
        io_threads = []
        for name in "out", "err":
            proc_stream = getattr(proc, "std" + name)
            tgt, args = self._methods[name]
            thr = threading.Thread(target=tgt, args=[proc_stream] + list(args))
            io_threads.append(thr)
        thr_proc = threading.Thread(target=self._run_process, args=(proc,))

        for thr in io_threads:
            thr.start()
        _log.debug(f"IO-threads.begin")

        _log.debug(f"Process.begin")
        thr_proc.start()
        thr_proc.join()
        _log.debug(f"Process.end")

        for thr in io_threads:
            thr.join()
        _log.debug(f"IO-threads.end")

    def _redirect(self, stream, dest, progress):
        step = 1
        for line in stream:
            s = line.decode("utf-8")
            dest.write(s)
            if self._delim and self._delim.search(s):
                progress.write(f"{step} ")
                progress.flush()
                step += 1
        stream.close()

    @staticmethod
    def _vanish(stream):
        for line in stream:
            pass
        stream.close()

    @staticmethod
    def _run_process(process):
        process.wait()


class ManagedWorkflow:
    """Manage a workflow of actions on datasets.
    """

    def __init__(self, name, workspace_path: Union[str, Path], tag=None):
        self._name = name
        workspace_path = Path(workspace_path)
        self._dmf = DMF(workspace_path, create=True)
        self._workspace_name = workspace_path.name
        self._tags = [] if tag is None else [tag]

    @property
    def name(self):
        return self._name

    @property
    def dmf(self):
        return self._dmf

    @property
    def workspace_name(self):
        return self._workspace_name

    @property
    def tags(self):
        return self._tags.copy()

    def get_dataset(self, type_, **kwargs):
        """Creates and returns a dataset of the specified type. If called more than once with the
        same type of dataset, then returns the previous value.
        """
        existing = self._dmf.find_one(name=type_)
        if existing:
            print(f"Already have an existing resource of type '{type_}'")
            return Dataset.from_resource(existing)
        dsf = DatasetFactory(type_, workflow=self)
        ds = dsf.create(target_path=self._download_path(), **kwargs)
        self._add_to_dmf(ds)
        return ds

    def _download_path(self):
        return self._dmf.workspace_path / "downloads"

    def _add_to_dmf(self, ds):
        datafile_list, datafiles_dir = [], ""
        if "files" in ds.meta:
            datafile_list = [
                {"path": filename, "desc": f"{DatasetType.RTS_GMLC} file {filename}"}
                for filename in ds.meta["files"]
            ]
        if "directory" in ds.meta:
            datafiles_dir = str(Path(ds.meta["directory"]).resolve())
        r = resource.Resource(
            {"datafiles": datafile_list, "datafiles_dir": datafiles_dir,
             "tags": self._tags}
        )
        r.set_field("name", ds.name)
        self._dmf.add(r)
        ds.resource = r

    def run_script(
        self,
        filename,
        collector: OutputCollector = None,
        output_dirs: List[Tuple[Path, str]] = None,
    ):
        """Run a downloaded script.

        The script is recorded as the DMF input resource.
        """
        if output_dirs is None:
            output_dirs = []
        # Add script to DMF
        path = self._download_path() / filename
        dsf = DatasetFactory("script", workflow=self)
        ds = dsf.create(path=path)
        resources = list(self._dmf.find(name=ds.name))
        if len(resources) == 0:
            _log.debug(f"Adding script configuration to DMF: {path}")
            self._add_to_dmf(ds)
            skip_run_check = True
        elif len(resources) == 1:
            _log.debug(f"Script configuration already in DMF: {path}")
            ds.resource = resources[0]
            skip_run_check = False
        else:
            n = len(resources)
            raise ValueError(
                f"Got {n} resources for script configuration, expected at most one: {path}"
            )
        # make all output directories into Path objects
        output_dirs = [(Path(o), g) for o, g in output_dirs]
        # run script through the 'run' method
        return self.run(
            rts_gmlc.runner,
            inputs=[ds],
            skip_check=skip_run_check,
            collector=collector,
            output_dirs=output_dirs,
            desc=f"Script {filename}"
        )

    def run(self, method, inputs=None, skip_check=False,
            desc=None, *args, **kwargs):
        """Run a processing step.

        Returns:
            (step-resource, output-resources): where step-resource is
              a DMF Resource created for the processing step, and output-resources
              is a list of DMF Resources, which may be empty, for the
              outputs
        """
        step_name = f"{method.__module__}.{method.__name__}"
        _log.debug(f"step.start name={step_name}")
        # Normalize inputs to a list, unless None
        if inputs is not None and not hasattr(inputs, "__iter__"):
            inputs = [inputs]
        # Check if processing step + inputs is in the DMF
        if not skip_check:
            existing = False
            if inputs is None:
                input_ids = None
            else:
                input_ids = {i.resource_id for i in inputs}
            for proc_step in self._dmf.find(name=step_name):
                proc_step_input_ids = set()
                for (_d, rel, meta) in self._dmf.find_related(
                    proc_step, outgoing=True, maxdepth=1
                ):
                    if rel.predicate == resource.Predicates.uses:
                        rid = meta[resource.Resource.ID_FIELD]
                        proc_step_input_ids.add(rid)
                if (input_ids is None and len(proc_step_input_ids) == 0) or (
                    input_ids is not None and proc_step_input_ids == input_ids
                ):
                    existing = True
                    break
            if existing:
                if input_ids is None:
                    _log.info(f"Step {step_name} has already been run (with no inputs)")
                else:
                    _log.info(
                        f"Step {step_name} has already been run with given inputs"
                    )
                _log.debug(f"step.end name={step_name} duplicate")
                return
        # set up arguments
        if inputs is not None:
            # prepend
            args = list(args)
            args.insert(0, inputs)
        # run the method
        _log.debug(f"step.run.start name={step_name}")
        outputs = method(*args, **kwargs)
        _log.debug(f"step.run.end name={step_name}")
        # Add processing step to DMF
        if desc is None:
            desc = f"Processing step {step_name}"
        step_resource = resource.Resource({"desc": desc, "tags": self._tags})
        step_resource.set_field("name", step_name)
        self._dmf.add(step_resource)
        # Link processing step resource to input dataset(s)
        if inputs is not None:
            for input_ in inputs:
                _log.debug(f"Link input {input_.name} to processing step")
                self._add_input_relation(step=step_resource, input_=input_.resource)
        # Create resource for output dataset
        output_resources = []
        if outputs:
            # loop over all types of outputs
            for key, value in outputs.items():
                # loop over directories of files for each output type
                for dir_path, files in value:
                    datafile_list = [
                        {
                            "path": str(filename),
                            "desc": f"{DatasetType.RTS_GMLC} file {filename}",
                        }
                        for filename in files
                    ]
                    output_resource = resource.Resource(
                        {
                            "desc": f"Output files for processing step {step_name}",
                            "datafiles": datafile_list,
                            "datafiles_dir": str(dir_path),
                            "tags": self._tags,
                        }
                    )
                    output_resource.set_field("name", key)
                    if _log.isEnabledFor(logging.DEBUG):
                        file_list = ", ".join([str(f) for f in files])
                        _log.debug(f"Add resource for output files: {file_list}")
                    self._dmf.add(output_resource)
                    # Link processing step resource to output dataset
                    self._add_output_relation(
                        step=step_resource, output=output_resource
                    )
                    _log.debug("Link output files to processing step")
                    output_resources.append(output_resource)
        # Push relations into DMF
        _log.debug("Push all relations into DMF with .update()")
        self._dmf.update()
        _log.debug(f"step.end name={step_name}")
        return step_resource, output_resources

    @staticmethod
    def _add_input_relation(step, input_):
        resource.create_relation(step, resource.Predicates.derived, input_)

    @staticmethod
    def _add_output_relation(step, output):
        resource.create_relation(output, resource.Predicates.derived, step)
