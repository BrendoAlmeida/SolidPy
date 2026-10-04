import pytest

from solidpy import (
    V12_NUMERICAL_ACCEPTANCE_POLICY,
    evaluate_numerical_acceptance,
)
from solidpy.backends._tolerances import TOLERANCES_VERSION
from solidpy.batch.assemble import kernel_source_hash
from solidpy.provenance import REFERENCE_PHYSICS_EQUIVALENCE_CLASS


def _result(scale=1.0, *, completed=True, mass_balance=0.2):
    return {
        "status": {
            "completed": completed,
            "numerical_blowdown_completed": completed,
        },
        "provenance": {
            "eta_c_applied": 1.0,
            "eta_cf_applied": 1.0,
            "discharge_coefficient_applied": 1.0,
            "efficiency_semantics": "native_split",
            "cea_used": False,
            "physics_provider_hash": "provider-a",
            "solidpy_git_sha": "commit-a",
        },
        "metrics": {
            "peak_chamber_pressure_pa": 5e6 * scale,
            "peak_thrust_n": 5000.0 * scale,
            "max_generated_mass_flow_kg_s": 2.0 * scale,
            "max_nozzle_mass_flow_kg_s": 1.8 * scale,
            "total_impulse_ns": 1000.0 * scale,
            "generated_mass_integral_kg": 2.0 * scale,
            "nozzle_mass_integral_kg": 1.9 * scale,
            "mass_flow_balance_error_pct": mass_balance,
        },
    }


def _certified_backend_result(
    *, backend="jax", equivalence_class=REFERENCE_PHYSICS_EQUIVALENCE_CLASS, certificate=True,
):
    result = _result()
    kernel_hash = kernel_source_hash()
    execution = {
        "backend": backend,
        "kernel_source_hash": kernel_hash,
        "tolerances_version": TOLERANCES_VERSION,
        "physics_equivalence_class": equivalence_class,
    }
    if certificate:
        execution["parity_certificate"] = {
            "suite": "solidpy-backend-parity",
            "suite_version": "1",
            "physics_equivalence_class": equivalence_class,
            "kernel_source_hash": kernel_hash,
            "tolerances_version": TOLERANCES_VERSION,
            "passed": True,
        }
    result["provenance"]["execution"] = execution
    return result


def test_v12_numerical_acceptance_passes_matching_completed_runs():
    report = evaluate_numerical_acceptance(_result(), _result(1.005))

    assert report["policy_id"] == V12_NUMERICAL_ACCEPTANCE_POLICY
    assert report["status"] == "passed"
    assert report["passed"] is True
    assert report["mass_balance_passed"] is True
    assert report["convergence"]["peak_thrust_n"]["relative_delta"] == pytest.approx(
        0.005 / 1.005
    )
    assert report["provenance"]["coarse"]["physics_equivalence_class"] == REFERENCE_PHYSICS_EQUIVALENCE_CLASS


def test_v12_numerical_acceptance_accepts_certified_backend_against_legacy_reference():
    report = evaluate_numerical_acceptance(_certified_backend_result(), _result(1.005))

    assert report["status"] == "passed"
    assert report["incomplete_reasons"] == []
    assert report["provenance"]["coarse"]["physics_equivalence_class"] == REFERENCE_PHYSICS_EQUIVALENCE_CLASS
    assert report["provenance"]["refined"]["physics_equivalence_class"] == REFERENCE_PHYSICS_EQUIVALENCE_CLASS


def test_v12_numerical_acceptance_rejects_different_certified_physics_classes():
    report = evaluate_numerical_acceptance(
        _certified_backend_result(backend="third-party", equivalence_class="solidpy-experimental-v1"), _result()
    )

    assert report["status"] == "incomplete"
    assert "physics_equivalence_class_mismatch" in report["incomplete_reasons"]


def test_v12_numerical_acceptance_requires_a_valid_backend_parity_certificate():
    uncertified = _certified_backend_result(certificate=False)

    report = evaluate_numerical_acceptance(uncertified, _result())

    assert report["status"] == "incomplete"
    assert "physics_equivalence_class_uncertified" in report["incomplete_reasons"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("kernel_source_hash", "b" * 64), ("tolerances_version", "stale"), ("suite_version", "2")),
)
def test_v12_numerical_acceptance_rejects_stale_parity_certificates(field, value):
    result = _certified_backend_result()
    result["provenance"]["execution"]["parity_certificate"][field] = value

    report = evaluate_numerical_acceptance(result, _result())

    assert report["status"] == "incomplete"
    assert "physics_equivalence_class_uncertified" in report["incomplete_reasons"]


def test_v12_numerical_acceptance_rejects_an_unvalidated_builtin_kernel_hash():
    result = _certified_backend_result()
    stale_hash = "b" * 64
    result["provenance"]["execution"]["kernel_source_hash"] = stale_hash
    result["provenance"]["execution"]["parity_certificate"]["kernel_source_hash"] = stale_hash

    report = evaluate_numerical_acceptance(result, _result())

    assert report["status"] == "incomplete"
    assert "physics_equivalence_class_uncertified" in report["incomplete_reasons"]


def test_v12_numerical_acceptance_requires_an_explicit_class_for_executed_backends():
    uncertified = _certified_backend_result()
    del uncertified["provenance"]["execution"]["physics_equivalence_class"]

    report = evaluate_numerical_acceptance(uncertified, _result())

    assert report["status"] == "incomplete"
    assert "physics_equivalence_class_missing" in report["incomplete_reasons"]


def test_v12_numerical_acceptance_fails_mass_balance_and_convergence_independently():
    report = evaluate_numerical_acceptance(
        _result(mass_balance=1.1), _result(1.04, mass_balance=0.5)
    )

    assert report["status"] == "failed"
    assert report["mass_balance_passed"] is False
    assert report["convergence"]["peak_thrust_n"]["passed"] is False


def test_v12_numerical_acceptance_marks_missing_metrics_incomplete():
    coarse = _result()
    del coarse["metrics"]["total_impulse_ns"]

    report = evaluate_numerical_acceptance(coarse, _result())

    assert report["status"] == "incomplete"
    assert report["passed"] is False
    assert "total_impulse_ns" in report["missing_metrics"]


def test_v12_numerical_acceptance_requires_numerical_blowdown():
    report = evaluate_numerical_acceptance(_result(completed=False), _result())

    assert report["status"] == "incomplete"
    assert "numerical_blowdown_not_completed" in report["incomplete_reasons"]


def test_v12_numerical_acceptance_rejects_incomparable_provenance():
    refined = _result()
    refined["provenance"]["physics_provider_hash"] = "different-inputs"
    refined["provenance"]["eta_cf_applied"] = 0.9

    report = evaluate_numerical_acceptance(_result(), refined)

    assert report["status"] == "incomplete"
    assert "physical_inputs_or_provider_mismatch" in report["incomplete_reasons"]
    assert "applied_efficiencies_mismatch" in report["incomplete_reasons"]


def test_v12_numerical_acceptance_validates_policy_and_scale_floors():
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_numerical_acceptance(_result(), _result(), policy_id="unknown")
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_numerical_acceptance(
            _result(), _result(), scale_floors={"peak_thrust_n": 0.0}
        )
