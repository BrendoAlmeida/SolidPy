import importlib.util
import os

import pytest

from solidpy import backends


def _accelerator_status():
    """Return ``(available, reason)``. JAX is only imported when a gpu test exists."""
    if importlib.util.find_spec("jax") is None:
        return False, "jax is not installed (install solidpy[jax-cuda12])"
    try:
        import jax

        devices = jax.devices()
    except Exception as exc:  # a broken CUDA install must be visible, not look like a missing package
        return False, f"jax is installed but failed to initialise a device: {type(exc).__name__}: {exc}"
    if any(device.platform != "cpu" for device in devices):
        return True, ""
    return False, "jax only sees CPU devices"


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.gpu`` tests when no accelerator device is present.

    Set ``SOLIDPY_REQUIRE_GPU=1`` (for a GPU runner) to turn that skip into an error, so a broken device
    cannot make the gpu tests silently disappear.
    """
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items:
        return
    available, reason = _accelerator_status()
    if available:
        return
    if os.environ.get("SOLIDPY_REQUIRE_GPU"):
        raise pytest.UsageError(f"SOLIDPY_REQUIRE_GPU is set but no accelerator is usable: {reason}")
    skip = pytest.mark.skip(reason=f"no accelerator device: {reason}")
    for item in gpu_items:
        item.add_marker(skip)


def pytest_generate_tests(metafunc):
    """Parametrize a test over every registered backend through the ``backend`` argument.

    The test receives the backend name. Backends whose library is missing show up as skipped with the
    install command, so a CPU-only run reports what it did not exercise.
    """
    if "backend" not in metafunc.fixturenames:
        return
    params = []
    for name, status in backends.available().items():
        if status == "ok":
            params.append(name)
        else:
            params.append(pytest.param(name, marks=pytest.mark.skip(reason=status)))
    metafunc.parametrize("backend", params)
