import logging
import sys
# expose selected objects in package namespace
from .workflow import ManagedWorkflow, DatasetType, OutputCollector


def set_log_level(level, dest=None, propagate=False):
    from .workflow import _log
    _log.setLevel(level)
    if dest:
        if isinstance(dest, int):
            if dest == 1:
                h = logging.StreamHandler(stream=sys.stdout)
            elif dest == 2:
                h = logging.StreamHandler()
            else:
                raise ValueError(f"Integer destination must be 1=stdout or 2=stderr. Got: {dest}")
        else:
            h = logging.FileHandler(filename=dest)
        h.setFormatter(logging.Formatter("%(asctime)s %(module)s %(levelname)s - %(message)s"))
        _log.addHandler(h)
    _log.propagate = propagate
