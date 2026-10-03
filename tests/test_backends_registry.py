import os
import subprocess
import sys
from pathlib import Path

import pytest

import solidpy
from solidpy import backends
from solidpy.backends import BackendUnavailable, Capabilities

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ENTRY_POINTS = backends._entry_points


class DummyBackend:
    name = "dummy"
    api_version = backends.BACKEND_API_VERSION

    def __init__(self, device=None):
        self.device = device

    def capabilities(self):
        return Capabilities({"tubular_grain": "supported", "star_grain": "partial"})

    def devices(self):
        return ["cpu"] if self.device is None else [self.device]

    def solve_burn(self, batch, options):
        raise NotImplementedError

    def provenance(self):
        return {"backend": self.name, "device": self.device}


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    """Isolate every test from registrations, cached instances and the selected backend."""
    monkeypatch.delenv(backends.ENV_BACKEND, raising=False)
    monkeypatch.delenv(backends.ENV_DEVICE, raising=False)
    monkeypatch.setattr(backends, "_REGISTERED", {})
    monkeypatch.setattr(backends, "_INSTANCES", {})
    monkeypatch.setattr(backends, "_process_default", None)
    monkeypatch.setattr(backends, "_entry_points", lambda: {})


def test_importing_solidpy_does_not_load_accelerator_libraries():
    code = (
        "import sys, solidpy; "
        "loaded = sorted(m for m in ('jax', 'jaxlib', 'torch', 'cupy', 'warp') if m in sys.modules); "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_the_default_backend_is_the_cpu_reference():
    assert backends.DEFAULT_BACKEND == "cpu-reference"
    assert backends.current_backend() == ("cpu-reference", None)


def test_registered_backend_is_listed_and_instantiated_once_per_device():
    backends.register_backend("dummy", DummyBackend)

    assert backends.available()["dummy"] == "ok"
    first = backends.get_backend("dummy")
    assert first is backends.get_backend("dummy")
    assert backends.get_backend("dummy", device="cuda:0").device == "cuda:0"
    assert backends.get_backend("dummy", device="cuda:0") is not first


def test_string_factory_is_imported_only_on_first_use():
    backends.register_backend("lazy", f"{__name__}:DummyBackend")
    assert backends.available()["lazy"] == "ok"
    assert isinstance(backends.get_backend("lazy"), DummyBackend)


def test_describe_reports_devices_capabilities_and_provenance():
    backends.register_backend("dummy", DummyBackend)

    description = backends.describe("dummy", device="cuda:0")

    assert description["name"] == "dummy"
    assert description["api_version"] == backends.BACKEND_API_VERSION
    assert description["devices"] == ["cuda:0"]
    assert description["capabilities"] == {"tubular_grain": "supported", "star_grain": "partial"}
    assert description["provenance"] == {"backend": "dummy", "device": "cuda:0"}


def test_missing_library_gives_an_actionable_error_not_an_import_crash():
    hint = 'pip install "solidpy[jax-cuda12]"'
    backends.register_backend(
        "needs-lib", DummyBackend, requires=("solidpy_library_that_does_not_exist",), install_hint=hint
    )

    assert backends.available()["needs-lib"] == f"missing: {hint}"
    with pytest.raises(ImportError, match=r"pip install \"solidpy\[jax-cuda12\]\""):
        backends.get_backend("needs-lib")
    with pytest.raises(BackendUnavailable, match="solidpy_library_that_does_not_exist"):
        backends.set_backend("needs-lib")
    with pytest.raises(BackendUnavailable):
        with backends.use_backend("needs-lib"):
            pass
    assert backends.current_backend() == ("cpu-reference", None)


def test_unknown_backend_lists_the_registered_names():
    backends.register_backend("dummy", DummyBackend)

    with pytest.raises(BackendUnavailable, match="unknown backend 'nope'.*dummy"):
        backends.get_backend("nope")


def test_duplicate_registration_needs_replace_and_unregister_forgets_instances():
    backends.register_backend("dummy", DummyBackend)
    first = backends.get_backend("dummy")

    with pytest.raises(ValueError, match="already registered"):
        backends.register_backend("dummy", DummyBackend)
    backends.register_backend("dummy", DummyBackend, replace=True)
    assert backends.get_backend("dummy") is not first

    backends.unregister_backend("dummy")
    assert "dummy" not in backends.available()
    with pytest.raises(KeyError):
        backends.unregister_backend("dummy")


def test_selection_precedence_scoped_then_process_then_environment_then_default(monkeypatch):
    for name in ("env", "process", "scoped"):
        backends.register_backend(name, DummyBackend)

    monkeypatch.setenv(backends.ENV_BACKEND, "env")
    monkeypatch.setenv(backends.ENV_DEVICE, "cuda:1")
    assert backends.current_backend() == ("env", "cuda:1")

    backends.set_backend("process", device="cuda:0")
    assert backends.current_backend() == ("process", "cuda:0")
    assert backends.get_backend().device == "cuda:0"

    with backends.use_backend("scoped"):
        assert backends.current_backend() == ("scoped", None)
        with backends.use_backend("env", device="cpu"):
            assert backends.current_backend() == ("env", "cpu")
        assert backends.current_backend() == ("scoped", None)
    assert backends.current_backend() == ("process", "cuda:0")

    backends.reset_backend()
    assert backends.current_backend() == ("env", "cuda:1")
    monkeypatch.delenv(backends.ENV_BACKEND)
    assert backends.current_backend() == ("cpu-reference", None)


def test_environment_device_applies_to_the_backend_the_environment_names(monkeypatch):
    for name in ("jaxlike", "cpulike"):
        backends.register_backend(name, DummyBackend)
    monkeypatch.setenv(backends.ENV_BACKEND, "jaxlike")
    monkeypatch.setenv(backends.ENV_DEVICE, "cuda:1")

    backends.set_backend("jaxlike")
    assert backends.current_backend() == ("jaxlike", "cuda:1")
    assert backends.get_backend().device == "cuda:1"
    assert backends.get_backend("jaxlike").device == "cuda:1"
    assert backends.get_backend("jaxlike", device="cuda:0").device == "cuda:0"

    with backends.use_backend("cpulike"):
        assert backends.current_backend() == ("cpulike", None)
        assert backends.get_backend().device is None

    monkeypatch.delenv(backends.ENV_BACKEND)
    backends.reset_backend()
    assert backends.current_backend() == ("cpu-reference", None)


def test_scoped_selection_is_restored_when_the_block_raises():
    backends.register_backend("scoped", DummyBackend)

    with pytest.raises(RuntimeError):
        with backends.use_backend("scoped"):
            raise RuntimeError("boom")

    assert backends.current_backend() == ("cpu-reference", None)


def test_package_level_helpers_are_the_registry_functions():
    assert solidpy.set_backend is backends.set_backend
    assert solidpy.use_backend is backends.use_backend
    assert solidpy.backends is backends


def test_third_party_backends_are_discovered_through_entry_points(monkeypatch):
    class FakeEntryPoint:
        name = "third-party"

        @staticmethod
        def load():
            return DummyBackend

    monkeypatch.setattr(backends, "_entry_points", lambda: {"third-party": FakeEntryPoint})

    assert backends.available()["third-party"] == "ok"
    assert isinstance(backends.get_backend("third-party"), DummyBackend)


def test_entry_point_that_cannot_be_imported_raises_backend_unavailable(monkeypatch):
    class BrokenEntryPoint:
        name = "broken"

        @staticmethod
        def load():
            raise ImportError("No module named 'vendor_sdk'")

    monkeypatch.setattr(backends, "_entry_points", lambda: {"broken": BrokenEntryPoint})

    with pytest.raises(BackendUnavailable, match="'broken' could not be loaded.*vendor_sdk"):
        backends.get_backend("broken")


def test_import_error_inside_a_registered_backend_becomes_backend_unavailable():
    def factory(device=None):
        raise ImportError("libcuda.so.1: cannot open shared object file")

    backends.register_backend("cuda-like", factory, install_hint="install the CUDA driver")

    with pytest.raises(BackendUnavailable, match="libcuda.*install the CUDA driver"):
        backends.get_backend("cuda-like")


def test_replacing_a_backend_while_it_is_being_instantiated_is_not_ignored():
    class Replacement(DummyBackend):
        name = "replacement"

    def slow_factory(device=None):
        # another thread re-registers the backend while this instance is still being created
        backends.register_backend("swap", Replacement, replace=True)
        return DummyBackend(device)

    backends.register_backend("swap", slow_factory)

    assert isinstance(backends.get_backend("swap"), Replacement)
    assert isinstance(backends.get_backend("swap"), Replacement)


def test_broken_entry_point_metadata_does_not_break_the_registry(monkeypatch):
    from importlib import metadata

    monkeypatch.setattr(backends, "_entry_points", REAL_ENTRY_POINTS)
    monkeypatch.setattr(metadata, "entry_points", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")))

    assert isinstance(backends.available(), dict)


def test_capabilities_never_assume_support_for_unlisted_features():
    capabilities = Capabilities({"tubular_grain": "supported", "star_grain": "partial"})

    assert capabilities.supports("tubular_grain")
    assert not capabilities.supports("star_grain")
    assert capabilities.level("igniter_callable") == "unsupported"
    assert capabilities.missing(["tubular_grain", "star_grain", "igniter_callable"]) == [
        "star_grain",
        "igniter_callable",
    ]
    with pytest.raises(ValueError, match="must be one of"):
        Capabilities({"tubular_grain": "yes"})
