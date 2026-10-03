# -*- coding: utf-8 -*-
"""Execution backends for batched simulations: registry and selection.

Nothing here imports an accelerator library. A backend is looked up by name and its module is imported
only when it is first requested, so ``import solidpy`` stays as light as before and a missing optional
library produces an actionable ``BackendUnavailable`` instead of an import-time crash.

Which backend runs is resolved in this order: a ``use_backend`` block, then ``set_backend``, then the
``SOLIDPY_BACKEND`` / ``SOLIDPY_DEVICE`` environment variables, then ``"cpu-reference"``.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib
import importlib.util
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

from ._protocol import (
    BACKEND_API_VERSION,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    Backend,
    BackendUnavailable,
    Capabilities,
    SolveOptions,
)

DEFAULT_BACKEND = "cpu-reference"
ENTRY_POINT_GROUP = "solidpy.backends"
ENV_BACKEND = "SOLIDPY_BACKEND"
ENV_DEVICE = "SOLIDPY_DEVICE"

Factory = Union[str, Callable[..., Backend]]


@dataclass(frozen=True)
class _Registration:
    name: str
    factory: Factory  # a callable ``factory(device=None)`` or a ``"module:Class"`` string
    requires: Tuple[str, ...] = ()  # top-level modules that must be importable
    install_hint: Optional[str] = None


# First-party backends, registered in code and imported lazily. Entry points are only for third parties.
_BUILTIN: Dict[str, _Registration] = {
    "cpu-reference": _Registration("cpu-reference", "solidpy.backends.cpu_reference:ReferenceBackend"),
}
_REGISTERED: Dict[str, _Registration] = {}
_INSTANCES: Dict[Tuple[str, Optional[str]], Backend] = {}
_LOCK = threading.RLock()
_GENERATION = 0  # bumped when a registration changes, so a slow instantiation can tell it went stale

_process_default: Optional[Tuple[str, Optional[str]]] = None
_scoped: "contextvars.ContextVar[Optional[Tuple[str, Optional[str]]]]" = contextvars.ContextVar(
    "solidpy_backend", default=None
)


def register_backend(
    name: str,
    factory: Factory,
    *,
    requires: Tuple[str, ...] = (),
    install_hint: Optional[str] = None,
    replace: bool = False,
) -> None:
    """Register a backend under ``name``.

    ``factory`` is a callable ``factory(device=None) -> Backend`` or a ``"package.module:Class"`` string
    that is imported on first use. ``requires`` lists the top-level modules the backend needs; they are
    checked without importing them.
    """
    global _GENERATION
    if not isinstance(name, str) or not name:
        raise ValueError("backend name must be a non-empty string")
    with _LOCK:
        if not replace and (name in _BUILTIN or name in _REGISTERED):
            raise ValueError(f"backend {name!r} is already registered")
        _REGISTERED[name] = _Registration(name, factory, tuple(requires), install_hint)
        _GENERATION += 1
        _drop_instances(name)


def unregister_backend(name: str) -> None:
    """Remove a backend registered with :func:`register_backend`. Built-in backends cannot be removed."""
    global _GENERATION
    with _LOCK:
        if name not in _REGISTERED:
            raise KeyError(f"backend {name!r} was not registered with register_backend")
        del _REGISTERED[name]
        _GENERATION += 1
        _drop_instances(name)


def _drop_instances(name: str) -> None:
    for key in [key for key in _INSTANCES if key[0] == name]:
        del _INSTANCES[key]


def _entry_points() -> Dict[str, Any]:
    """Third-party backends advertised through the ``solidpy.backends`` entry-point group."""
    try:
        from importlib import metadata

        found = metadata.entry_points()
        group = found.select(group=ENTRY_POINT_GROUP) if hasattr(found, "select") else found.get(ENTRY_POINT_GROUP, [])
        return {entry.name: entry for entry in group}
    except Exception:  # broken third-party metadata must not break solidpy
        return {}


def _resolve(name: str) -> Tuple[_Registration, int]:
    """Return ``(registration, generation)`` for ``name``, or raise ``BackendUnavailable`` listing the known names."""
    with _LOCK:
        registration = _REGISTERED.get(name) or _BUILTIN.get(name)
        generation = _GENERATION
        known = {**_BUILTIN, **_REGISTERED}
    if registration is not None:
        return registration, generation
    entries = _entry_points()  # scanned once, outside the lock
    entry = entries.get(name)
    if entry is not None:
        return _Registration(name, lambda device=None, _entry=entry: _entry.load()(device=device)), generation
    names = ", ".join(sorted({*known, *entries})) or "none"
    raise BackendUnavailable(f"unknown backend {name!r}; registered backends: {names}")


def _missing_requirements(registration: _Registration) -> list:
    return [module for module in registration.requires if importlib.util.find_spec(module) is None]


def _unavailable_message(registration: _Registration, missing: list) -> str:
    message = f"backend {registration.name!r} needs the missing package(s): {', '.join(missing)}."
    if registration.install_hint:
        message += f" Install it with: {registration.install_hint}"
    return message


def available() -> Dict[str, str]:
    """Return ``{name: "ok" | "missing: <how to install>"}`` for every known backend.

    Availability of the libraries is checked without importing them. Backends advertised through entry
    points are listed as ``"ok"`` without being loaded; loading one that is broken raises ``BackendUnavailable``.
    """
    with _LOCK:
        registrations = {**_BUILTIN, **_REGISTERED}
    status: Dict[str, str] = {}
    for name, registration in sorted(registrations.items()):
        missing = _missing_requirements(registration)
        if missing:
            hint = registration.install_hint or ", ".join(missing)
            status[name] = f"missing: {hint}"
        else:
            status[name] = "ok"
    for name in _entry_points():
        status.setdefault(name, "ok")
    return status


def _instantiate(registration: _Registration, device: Optional[str]) -> Backend:
    missing = _missing_requirements(registration)
    if missing:
        raise BackendUnavailable(_unavailable_message(registration, missing))
    try:
        factory = registration.factory
        if isinstance(factory, str):
            module_name, _, attribute = factory.partition(":")
            factory = getattr(importlib.import_module(module_name), attribute)
        return factory(device=device)
    except BackendUnavailable:
        raise
    except ImportError as exc:
        message = f"backend {registration.name!r} could not be loaded: {exc}."
        if registration.install_hint:
            message += f" Install it with: {registration.install_hint}"
        raise BackendUnavailable(message) from exc


def get_backend(name: Optional[str] = None, device: Optional[str] = None) -> Backend:
    """Return the backend instance for ``name`` (default: the one selected for the current context)."""
    if name is None:
        name, selected = current_backend()
        device = selected if device is None else device
    else:
        device = _resolve_device(name, device)
    key = (name, device)
    while True:
        registration, generation = _resolve(name)
        with _LOCK:
            instance = _INSTANCES.get(key)
        if instance is not None:
            return instance
        candidate = _instantiate(registration, device)  # outside the lock: importing a library can be slow
        with _LOCK:
            if generation == _GENERATION:
                return _INSTANCES.setdefault(key, candidate)
        # the registry changed while this backend was being created: resolve again


def describe(name: str, device: Optional[str] = None) -> Dict[str, Any]:
    """Describe a backend: version, devices, capability matrix and provenance fragment.

    This instantiates the backend, so for accelerator backends it imports the library.
    """
    backend = get_backend(name, device)
    capabilities = backend.capabilities()
    return {
        "name": backend.name,
        "api_version": backend.api_version,
        "devices": list(backend.devices()),
        "capabilities": dict(capabilities.features),
        "dtypes": list(capabilities.dtypes),
        "history_policies": list(capabilities.history_policies),
        "provenance": backend.provenance(),
    }


def _resolve_device(name: str, device: Optional[str]) -> Optional[str]:
    """An explicit device wins; otherwise ``SOLIDPY_DEVICE`` applies to the backend ``SOLIDPY_BACKEND`` names."""
    if device is not None:
        return device
    if name == os.environ.get(ENV_BACKEND):
        return os.environ.get(ENV_DEVICE) or None
    return None


def current_backend() -> Tuple[str, Optional[str]]:
    """Return the ``(name, device)`` selected for the current context, without instantiating it."""
    scoped = _scoped.get()
    if scoped is not None:
        name, device = scoped
    elif _process_default is not None:
        name, device = _process_default
    else:
        name, device = os.environ.get(ENV_BACKEND) or DEFAULT_BACKEND, None
    return name, _resolve_device(name, device)


def _validated(name: str, device: Optional[str]) -> Tuple[str, Optional[str]]:
    registration, _ = _resolve(name)
    missing = _missing_requirements(registration)
    if missing:
        raise BackendUnavailable(_unavailable_message(registration, missing))
    return name, device


def set_backend(name: str, device: Optional[str] = None) -> None:
    """Select the process-wide default backend. The default itself stays ``"cpu-reference"``."""
    global _process_default
    _process_default = _validated(name, device)


def reset_backend() -> None:
    """Forget :func:`set_backend` and fall back to the environment variables or ``"cpu-reference"``."""
    global _process_default
    _process_default = None


@contextlib.contextmanager
def use_backend(name: str, device: Optional[str] = None) -> Iterator[Tuple[str, Optional[str]]]:
    """Scoped override of the backend (thread and async safe)."""
    selection = _validated(name, device)
    token = _scoped.set(selection)
    try:
        yield selection
    finally:
        _scoped.reset(token)


__all__ = [
    "BACKEND_API_VERSION",
    "DEFAULT_BACKEND",
    "PARTIAL",
    "SUPPORTED",
    "UNSUPPORTED",
    "Backend",
    "BackendUnavailable",
    "Capabilities",
    "SolveOptions",
    "available",
    "current_backend",
    "describe",
    "get_backend",
    "register_backend",
    "reset_backend",
    "set_backend",
    "unregister_backend",
    "use_backend",
]
