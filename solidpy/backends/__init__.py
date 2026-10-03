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
_BUILTIN: Dict[str, _Registration] = {}
_REGISTERED: Dict[str, _Registration] = {}
_INSTANCES: Dict[Tuple[str, Optional[str]], Backend] = {}
_LOCK = threading.RLock()

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
    if not isinstance(name, str) or not name:
        raise ValueError("backend name must be a non-empty string")
    with _LOCK:
        if not replace and (name in _BUILTIN or name in _REGISTERED):
            raise ValueError(f"backend {name!r} is already registered")
        _REGISTERED[name] = _Registration(name, factory, tuple(requires), install_hint)
        _drop_instances(name)


def unregister_backend(name: str) -> None:
    """Remove a backend registered with :func:`register_backend`. Built-in backends cannot be removed."""
    with _LOCK:
        if name not in _REGISTERED:
            raise KeyError(f"backend {name!r} was not registered with register_backend")
        del _REGISTERED[name]
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


def _lookup(name: str) -> _Registration:
    with _LOCK:
        registration = _REGISTERED.get(name) or _BUILTIN.get(name)
        if registration is not None:
            return registration
    entry = _entry_points().get(name)
    if entry is not None:
        return _Registration(name, lambda device=None, _entry=entry: _entry.load()(device=device))
    known = ", ".join(sorted(available())) or "none"
    raise BackendUnavailable(f"unknown backend {name!r}; registered backends: {known}")


def _missing_requirements(registration: _Registration) -> list:
    return [module for module in registration.requires if importlib.util.find_spec(module) is None]


def _unavailable_message(registration: _Registration, missing: list) -> str:
    message = f"backend {registration.name!r} needs the missing package(s): {', '.join(missing)}."
    if registration.install_hint:
        message += f" Install it with: {registration.install_hint}"
    return message


def available() -> Dict[str, str]:
    """Return ``{name: "ok" | "missing: <how to install>"}`` for every known backend.

    Availability of the libraries is checked without importing them.
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
    factory = registration.factory
    if isinstance(factory, str):
        module_name, _, attribute = factory.partition(":")
        factory = getattr(importlib.import_module(module_name), attribute)
    return factory(device=device)


def get_backend(name: Optional[str] = None, device: Optional[str] = None) -> Backend:
    """Return the backend instance for ``name`` (default: the one selected for the current context)."""
    if name is None:
        name, selected_device = current_backend()
        device = device if device is not None else selected_device
    registration = _lookup(name)
    with _LOCK:
        key = (name, device)
        instance = _INSTANCES.get(key)
        if instance is None:
            instance = _INSTANCES[key] = _instantiate(registration, device)
    return instance


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


def current_backend() -> Tuple[str, Optional[str]]:
    """Return the ``(name, device)`` selected for the current context, without instantiating it."""
    scoped = _scoped.get()
    if scoped is not None:
        return scoped
    if _process_default is not None:
        return _process_default
    name = os.environ.get(ENV_BACKEND)
    if name:
        return name, os.environ.get(ENV_DEVICE) or None
    return DEFAULT_BACKEND, None


def _validated(name: str, device: Optional[str]) -> Tuple[str, Optional[str]]:
    registration = _lookup(name)
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
