import subprocess
import sys


def test_a_missing_jax_is_reported_without_importing_it():
    code = r"""
import importlib.util
import sys

find_spec = importlib.util.find_spec
def without_jax(name, *args, **kwargs):
    return None if name == "jax" else find_spec(name, *args, **kwargs)

importlib.util.find_spec = without_jax
from solidpy import backends
from solidpy.backends import BackendUnavailable

assert backends.available()["jax"].startswith("missing: pip install")
assert "solidpy[jax-cuda12]" in backends.available()["jax"]
try:
    backends.get_backend("jax")
except BackendUnavailable as exc:
    assert "needs the missing package(s): jax" in str(exc)
    assert "solidpy[jax-cuda12]" in str(exc)
else:
    raise AssertionError("get_backend should reject JAX when its package is unavailable")
assert "jax" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
