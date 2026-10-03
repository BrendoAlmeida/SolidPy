"""Shared process-pool settings for host work that may follow accelerator execution."""

from __future__ import annotations

import importlib
import multiprocessing
import os
import pickle
import types

import numpy as np

MAX_PROCESS_WORKERS = 16


def available_cpu_count():
    """Return the CPUs available to this process, honoring affinity where the runtime exposes it."""
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return max(1, len(get_affinity(0)))
        except OSError:
            pass
    return os.cpu_count() or 1


def process_worker_count(requested, task_count):
    """Cap a requested process count to available CPUs, work items and a library-wide ceiling."""
    cpu_count = available_cpu_count()
    return max(1, min(int(requested), int(task_count), cpu_count, MAX_PROCESS_WORKERS))


def spawn_pickle_safe(value, *, skip_numeric_arrays=False):
    """Check that a value is pickleable and its referenced types can be imported by a spawned worker."""
    seen = set()

    def importable(target):
        module_name = getattr(target, "__module__", None)
        qualname = getattr(target, "__qualname__", None)
        if module_name in ("__main__", "__mp_main__") or (qualname and "<locals>" in qualname):
            return False
        if not module_name or module_name in ("builtins", "__builtin__") or not qualname:
            return True
        try:
            resolved = importlib.import_module(module_name)
            for component in qualname.split("."):
                resolved = getattr(resolved, component)
        except (ImportError, AttributeError, ValueError):
            return False
        return resolved is target

    def check(item):
        if isinstance(item, np.ndarray):
            if skip_numeric_arrays and type(item) is np.ndarray and not item.dtype.hasobject:
                return True
            if item.dtype.hasobject:
                if id(item) in seen:
                    return True
                seen.add(id(item))
                if not all(check(element) for element in item.flat):
                    return False
        if isinstance(item, dict):
            if id(item) in seen:
                return True
            seen.add(id(item))
            if not all(check(key) and check(value) for key, value in item.items()):
                return False
        elif isinstance(item, (list, tuple, set, frozenset)):
            if id(item) in seen:
                return True
            seen.add(id(item))
            if not all(check(element) for element in item):
                return False

        if isinstance(item, (types.MethodType, types.BuiltinMethodType)):
            if not check(item.__self__):
                return False
            target = item.__func__ if isinstance(item, types.MethodType) else item
        else:
            target = item
        if not isinstance(target, (types.FunctionType, types.BuiltinFunctionType, type)):
            target = type(target)
        if not importable(target):
            return False

        try:
            attributes = getattr(item, "__dict__", None)
        except Exception:
            return False
        if isinstance(attributes, dict) and id(item) not in seen:
            seen.add(id(item))
            if not check(attributes):
                return False
        for cls in type(item).__mro__:
            slots = getattr(cls, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in ("__dict__", "__weakref__") or not hasattr(item, slot):
                    continue
                try:
                    if not check(getattr(item, slot)):
                        return False
                except Exception:
                    return False
        try:
            pickle.dumps(item)
        except Exception:
            return False
        return True

    return check(value)


def safe_process_context():
    """Choose a fresh process context so workers cannot inherit initialized accelerator state."""
    return multiprocessing.get_context("spawn")


def process_worker_initializer():
    """Keep native math libraries from starting one thread per CPU inside each worker process."""
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=1)
