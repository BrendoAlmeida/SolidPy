"""Shared provenance rules for deciding when simulation results describe equivalent physics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Optional


REFERENCE_PHYSICS_EQUIVALENCE_CLASS = "solidpy-reference-v1"
_PARITY_CERTIFIED_BUILTIN_BACKENDS = frozenset({"cpu-vectorized", "jax"})
_PARITY_SUITE = "solidpy-backend-parity"
_PARITY_SUITE_VERSION = "1"
# Add a kernel/tolerance pair only after the parity suite passes on the supported device classes.
_CERTIFIED_KERNELS = frozenset({
    (
        "5",
        "54ce494f3fd3172f6caeb82b4856b7729fa92d3b927ca7c88275efc82f9daf03",
    ),
    (
        "5",
        "8cab0323f6e63a08770d8dd92d7f88f6140fb2b497de41f55d3e8f912219f6a0",
    ),
})


def builtin_parity_certificate(
    backend: Optional[str], kernel_source_hash: str, tolerances_version: str,
) -> Optional[Dict[str, Any]]:
    """Return the shipped parity certificate for a built-in accelerated backend, if one exists."""
    if (
        backend not in _PARITY_CERTIFIED_BUILTIN_BACKENDS
        or (str(tolerances_version), kernel_source_hash) not in _CERTIFIED_KERNELS
    ):
        return None
    return {
        "suite": _PARITY_SUITE,
        "suite_version": _PARITY_SUITE_VERSION,
        "physics_equivalence_class": REFERENCE_PHYSICS_EQUIVALENCE_CLASS,
        "kernel_source_hash": kernel_source_hash,
        "tolerances_version": str(tolerances_version),
        "passed": True,
    }


def unverified_physics_equivalence_class(backend: Optional[str], kernel_source_hash: str) -> str:
    """Give uncertified implementations a class that cannot be mixed with the reference class."""
    backend_id = backend if isinstance(backend, str) and backend else "unknown-backend"
    return f"solidpy-unverified-v1:{backend_id}:{kernel_source_hash}"


def result_physics_equivalence_class(provenance: Any) -> Optional[str]:
    """Return a result's certified physics class, treating legacy scalar results as the reference class.

    Scalar ``BurnSimulation`` results predate this field and remain byte-for-byte compatible. A result with no
    execution block is therefore the reference implementation. Batched or third-party results must carry an explicit
    class and a passing certificate bound to the kernel source recorded in their execution provenance.
    """
    if not isinstance(provenance, Mapping):
        return None
    if "execution" not in provenance:
        return REFERENCE_PHYSICS_EQUIVALENCE_CLASS
    execution = provenance.get("execution")
    if not isinstance(execution, Mapping):
        return None
    if execution.get("backend") == "cpu-reference":
        return REFERENCE_PHYSICS_EQUIVALENCE_CLASS

    equivalence_class = execution.get("physics_equivalence_class")
    certificate = execution.get("parity_certificate")
    if not isinstance(equivalence_class, str) or not equivalence_class:
        return None
    if not isinstance(certificate, Mapping) or certificate.get("passed") is not True:
        return None
    if certificate.get("physics_equivalence_class") != equivalence_class:
        return None
    if certificate.get("suite") != _PARITY_SUITE or certificate.get("suite_version") != _PARITY_SUITE_VERSION:
        return None
    kernel_hash = execution.get("kernel_source_hash")
    if not isinstance(kernel_hash, str) or not kernel_hash or certificate.get("kernel_source_hash") != kernel_hash:
        return None
    tolerances_version = execution.get("tolerances_version")
    certificate_tolerances = certificate.get("tolerances_version")
    if (
        not isinstance(tolerances_version, str)
        or not tolerances_version
        or certificate_tolerances != tolerances_version
    ):
        return None
    if execution.get("backend") in _PARITY_CERTIFIED_BUILTIN_BACKENDS and (
        equivalence_class != REFERENCE_PHYSICS_EQUIVALENCE_CLASS
        or (tolerances_version, kernel_hash) not in _CERTIFIED_KERNELS
    ):
        return None
    return equivalence_class
