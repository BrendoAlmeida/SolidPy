import pytest

from solidpy import (
    V12_NUMERICAL_ACCEPTANCE_POLICY,
    evaluate_numerical_acceptance,
)


def _result(scale=1.0, *, completed=True, mass_balance=0.2):
    return {
        "status": {
            "completed": completed,
            "numerical_blowdown_completed": completed,
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


def test_v12_numerical_acceptance_passes_matching_completed_runs():
    report = evaluate_numerical_acceptance(_result(), _result(1.005))

    assert report["policy_id"] == V12_NUMERICAL_ACCEPTANCE_POLICY
    assert report["status"] == "passed"
    assert report["passed"] is True
    assert report["mass_balance_passed"] is True
    assert report["convergence"]["peak_thrust_n"]["relative_delta"] == pytest.approx(
        0.005 / 1.005
    )


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


def test_v12_numerical_acceptance_validates_policy_and_scale_floors():
    with pytest.raises(ValueError, match="unsupported"):
        evaluate_numerical_acceptance(_result(), _result(), policy_id="unknown")
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_numerical_acceptance(
            _result(), _result(), scale_floors={"peak_thrust_n": 0.0}
        )
