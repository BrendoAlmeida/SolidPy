import importlib.util

import pytest

from solidpy import backends


def _accelerator_available():
    """True when JAX is installed and sees a non-CPU device. JAX is only imported if a gpu test exists."""
    if importlib.util.find_spec("jax") is None:
        return False
    try:
        import jax

        return any(device.platform != "cpu" for device in jax.devices())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.gpu`` tests cleanly when no accelerator device is present."""
    gpu_items = [item for item in items if item.get_closest_marker("gpu")]
    if not gpu_items or _accelerator_available():
        return
    skip = pytest.mark.skip(reason="no accelerator device (install solidpy[jax-cuda12] and use a CUDA GPU)")
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
