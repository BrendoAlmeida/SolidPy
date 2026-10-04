import math
from types import MappingProxyType

import numpy as np
import pytest

from solidpy import CasingMaterial, MotorGeometry, StructuralMonteCarlo, backends
from solidpy.backends import Capabilities
from solidpy.backends.numpy_vectorized import NumpyBackend


def make_geometry():
    return MotorGeometry(
        motor_length_m=0.24,
        motor_inner_diameter_m=0.08,
        casing_wall_thickness_m=0.004,
        grain_outer_diameter_m=0.07,
        grain_core_diameter_m=0.03,
        grain_gap_m=0.0,
        grain_length_each_m=0.10,
        grain_number=2,
        fill_length_m=0.20,
        throat_diameter_m=0.012,
        exit_diameter_m=0.028,
        free_volume_m3=0.001,
        propellant_mass_kg=1.0,
        dry_mass_kg=3.0,
        motor_initial_mass_kg=4.0,
        motor_final_mass_kg=3.0,
    )


def _model(distribution, *, seed=7, **kwargs):
    return StructuralMonteCarlo(
        make_geometry(), CasingMaterial(), peak_pressure_distribution=distribution,
        parameter_sigmas={"perturb_peak_pressure": {"mean": 1.0e5, "sigma": 2.0e4}},
        random_seed=seed, **kwargs,
    )


def _assert_structural_reports_match(reference, actual):
    numeric_series = {
        "burst_safety_factor", "governing_safety_factor", "bolt_shear_safety_factor",
        "bolt_bearing_safety_factor",
    }
    for name in (
        "status", "n_iterations", "n_evaluated", "n_failed", "failure_probability",
        "failure_probability_casing", "failure_probability_burst", "failure_probability_bolts",
        "closure_bolt_status", "closure_bolt_applicability", "closure_bolt_reason",
        "scenario_ids", "evaluated_scenario_ids", "samples", "peak_pressure_pa",
        "burst_safety_factor", "governing_safety_factor", "bolt_shear_safety_factor",
        "bolt_bearing_safety_factor",
    ):
        if name in numeric_series:
            assert len(actual[name]) == len(reference[name]), name
            for expected_value, actual_value in zip(reference[name], actual[name]):
                if expected_value is None:
                    assert actual_value is None, name
                else:
                    assert actual_value == pytest.approx(expected_value, rel=1e-12, abs=1e-12), name
        else:
            assert actual[name] == reference[name], name
    assert len(actual["scenarios"]) == len(reference["scenarios"])
    for expected, observed in zip(reference["scenarios"], actual["scenarios"]):
        assert (observed["scenario_id"], observed["status"]) == (
            expected["scenario_id"], expected["status"]
        )
        if expected["status"] == "failed":
            assert observed["error"] == expected["error"]
            continue
        assert observed["scenario_factors"] == expected["scenario_factors"]
        assert observed["peak_pressure_pa"] == pytest.approx(expected["peak_pressure_pa"], rel=1e-12)
        assert observed["structural"].keys() == expected["structural"].keys()
        for key, value in expected["structural"].items():
            actual_value = observed["structural"][key]
            if value is None or isinstance(value, str):
                assert actual_value == value, key
            else:
                assert math.isclose(actual_value, value, rel_tol=1e-12, abs_tol=1e-12), key


def test_cpu_vectorized_structural_monte_carlo_matches_scalar_reference():
    reference = _model(lambda: 8.0e6, bolt_count=4, bolt_diameter_m=0.006,
                       bolt_strength_mpa=800.0).run(128, backend="cpu-reference")
    vectorized = _model(lambda: 8.0e6, bolt_count=4, bolt_diameter_m=0.006,
                        bolt_strength_mpa=800.0).run(128, backend="cpu-vectorized")

    _assert_structural_reports_match(reference, vectorized)
    assert vectorized["provenance"]["execution"]["backend"] == "cpu-vectorized"
    assert vectorized["provenance"]["execution"]["service"] == "structural_response"
    assert vectorized["provenance"]["physics_provider_hash"] == reference["provenance"]["physics_provider_hash"]


def test_backend_none_keeps_the_legacy_reference_even_in_a_backend_context():
    model = _model(lambda: 3.0e6)
    with backends.use_backend("cpu-vectorized"):
        result = model.run(3)
    assert result["n_evaluated"] == 3
    assert "execution" not in result["provenance"]


def test_cpu_vectorized_structural_monte_carlo_isolates_invalid_pressure_lane():
    values = iter([3.0e6, np.nan, 5.0e6, 7.0e6])
    reference = _model(lambda: next(values), seed=11).run(4, backend="cpu-reference")
    values = iter([3.0e6, np.nan, 5.0e6, 7.0e6])
    vectorized = _model(lambda: next(values), seed=11).run(4, backend="cpu-vectorized")

    _assert_structural_reports_match(reference, vectorized)
    assert vectorized["n_failed"] == 1
    assert vectorized["scenarios"][1]["error"]["type"] == "ValueError"


def test_structural_monte_carlo_falls_back_for_backend_without_service():
    class NoStructuralService:
        name = "no-structural-service"
        api_version = backends.BACKEND_API_VERSION

        def __init__(self, device=None):
            self.device = device

        def capabilities(self):
            return Capabilities()

        def devices(self):
            return ["cpu"]

        def solve_burn(self, batch, options):
            raise NotImplementedError

        def provenance(self):
            return {"backend": self.name, "device": self.device}

    backends.register_backend("no-structural-service", NoStructuralService)
    try:
        result = _model(lambda: 3.0e6).run(3, backend="no-structural-service")
        assert result["n_evaluated"] == 3
        assert result["provenance"]["execution"]["fallback_lanes"] == [0, 1, 2]
        assert result["provenance"]["execution"]["fallback_reason"] == "service_not_supported"
        with pytest.raises(ValueError, match="does not provide"):
            _model(lambda: 3.0e6).run(1, backend="no-structural-service", strict=True)
    finally:
        backends.unregister_backend("no-structural-service")


def test_structural_response_service_accepts_any_mapping_implementation():
    class MappingBackend(NumpyBackend):
        name = "mapping-structural-service"

        def structural_response(self, *args, **kwargs):
            return MappingProxyType(super().structural_response(*args, **kwargs))

    backends.register_backend("mapping-structural-service", MappingBackend)
    try:
        result = _model(lambda: 3.0e6).run(
            3, backend="mapping-structural-service", strict=True,
        )
        assert result["n_evaluated"] == 3
        assert result["provenance"]["execution"]["fallback_lanes"] == []
    finally:
        backends.unregister_backend("mapping-structural-service")


def test_structural_response_requires_a_numeric_value_for_every_lane():
    class ScalarMetricsBackend(NumpyBackend):
        name = "scalar-structural-metrics"

        def structural_response(self, *args, **kwargs):
            output = super().structural_response(*args, **kwargs)
            return {
                name: (float(value[0]) if isinstance(value, np.ndarray) else value)
                for name, value in output.items()
            }

    backends.register_backend("scalar-structural-metrics", ScalarMetricsBackend)
    try:
        result = _model(lambda: 3.0e6).run(3, backend="scalar-structural-metrics")
        assert result["n_evaluated"] == 3
        assert result["provenance"]["execution"]["fallback_lanes"] == [0, 1, 2]
        assert "expected (3,)" in result["provenance"]["execution"]["fallback_reason"]
        with pytest.raises(ValueError, match=r"expected \(3,\)"):
            _model(lambda: 3.0e6).run(3, backend="scalar-structural-metrics", strict=True)
    finally:
        backends.unregister_backend("scalar-structural-metrics")


def test_invalid_metric_value_falls_back_only_for_its_lane():
    class OneBadMetricBackend(NumpyBackend):
        name = "one-bad-structural-metric"

        def structural_response(self, *args, **kwargs):
            output = super().structural_response(*args, **kwargs)
            values = np.asarray(
                output["simulation.advanced.structural.burst_safety_factor"], dtype=object
            )
            values[1] = "invalid"
            output["simulation.advanced.structural.burst_safety_factor"] = values
            return output

    backends.register_backend("one-bad-structural-metric", OneBadMetricBackend)
    try:
        result = _model(lambda: 4.0e6).run(3, backend="one-bad-structural-metric")
        assert result["n_evaluated"] == 3
        assert result["provenance"]["execution"]["fallback_lanes"] == [1]
        assert result["scenarios"][0]["status"] == "completed"
        assert result["scenarios"][2]["status"] == "completed"
    finally:
        backends.unregister_backend("one-bad-structural-metric")


def test_non_finite_numeric_metric_falls_back_only_for_its_lane():
    class OneNonFiniteMetricBackend(NumpyBackend):
        name = "one-non-finite-structural-metric"

        def structural_response(self, *args, **kwargs):
            output = super().structural_response(*args, **kwargs)
            key = "simulation.advanced.structural.burst_safety_factor"
            values = np.asarray(output[key]).copy()
            values[1] = np.nan
            output[key] = values
            return output

    backends.register_backend(OneNonFiniteMetricBackend.name, OneNonFiniteMetricBackend)
    try:
        reference = _model(lambda: 4.0e6).run(3, backend="cpu-reference")
        accelerated = _model(lambda: 4.0e6).run(3, backend=OneNonFiniteMetricBackend.name)

        _assert_structural_reports_match(reference, accelerated)
        assert accelerated["provenance"]["execution"]["fallback_lanes"] == [1]
        assert accelerated["provenance"]["execution"]["fallback_reason"] == "non_finite_metric"
        with pytest.raises(ValueError, match="contains a non-finite metric"):
            _model(lambda: 4.0e6).run(3, backend=OneNonFiniteMetricBackend.name, strict=True)
    finally:
        backends.unregister_backend(OneNonFiniteMetricBackend.name)


def test_jax_cpu_structural_monte_carlo_matches_scalar_reference():
    pytest.importorskip("jax")
    reference = _model(lambda: 8.0e6, bolt_count=4, bolt_diameter_m=0.006,
                       bolt_strength_mpa=800.0).run(32, backend="cpu-reference")
    accelerated = _model(lambda: 8.0e6, bolt_count=4, bolt_diameter_m=0.006,
                         bolt_strength_mpa=800.0).run(32, backend="jax", device="cpu")

    _assert_structural_reports_match(reference, accelerated)
    assert accelerated["provenance"]["execution"]["backend"] == "jax"
    assert accelerated["provenance"]["execution"]["kernel_source_hash"]


@pytest.mark.gpu
def test_jax_gpu_structural_monte_carlo_matches_scalar_reference():
    jax = pytest.importorskip("jax")
    try:
        if not jax.devices("gpu"):
            pytest.skip("JAX has no GPU device")
    except RuntimeError:
        pytest.skip("JAX has no GPU device")
    reference = _model(lambda: 8.0e6).run(32, backend="cpu-reference")
    accelerated = _model(lambda: 8.0e6).run(32, backend="jax", device="cuda:0")

    _assert_structural_reports_match(reference, accelerated)
    assert accelerated["provenance"]["execution"]["device"] == "cuda:0"
