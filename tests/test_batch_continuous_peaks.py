"""Continuous-peak checks for batched DOP853 results.

The scalar result records peaks on accepted points. These checks independently refine its DOP853 dense output so
the numerical-margin gate can compare the batched path with the continuous solution rather than either step grid.
"""

import importlib

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import golden_corpus as gc
from solidpy import BurnSimulation
from solidpy.batch.integrators import solver
from solidpy.ensemble import ProblemBatch, simulate_burn


def _refined_dense_peaks(case, monkeypatch):
    grain, motor, propellant, environment, options = gc.build_objects(case)
    options = dict(options, max_step_size=0.002, rtol=1e-11)
    burn_module = importlib.import_module("solidpy.Burn")
    original_solve_ivp = burn_module.solve_ivp
    dense_segments = []

    def solve_ivp_with_dense_output(*args, **kwargs):
        kwargs["dense_output"] = True
        result = original_solve_ivp(*args, **kwargs)
        dense_segments.append(result)
        return result

    monkeypatch.setattr(burn_module, "solve_ivp", solve_ivp_with_dense_output)
    try:
        simulation = BurnSimulation(grain, motor, propellant, environment, **options)
    finally:
        monkeypatch.setattr(burn_module, "solve_ivp", original_solve_ivp)
    metric_names = (
        "max_generated_mass_flow_kg_s", "peak_chamber_pressure_pa", "peak_thrust_n",
        "max_nozzle_mass_flow_kg_s",
    )
    maxima = {name: -np.inf for name in metric_names}
    cache = {}

    def quantities_at(time, segment):
        time = float(time)
        key = (id(segment), time)
        if key not in cache:
            cache[key] = simulation._state_quantities(time, segment.sol(time))
        return cache[key]

    def metric_value(q, name):
        if name == "max_generated_mass_flow_kg_s":
            return q["generated"]
        if name == "peak_chamber_pressure_pa":
            return q["pressure"]
        if name == "peak_thrust_n":
            return q["components"]["total_n"]
        return q["nozzle"]

    for segment in dense_segments:
        for left, right in zip(segment.t[:-1], segment.t[1:]):
            if right <= left:
                continue
            endpoints = (left, np.nextafter(right, -np.inf), right)
            for time in endpoints:
                q = quantities_at(time, segment)
                for name in metric_names:
                    maxima[name] = max(maxima[name], float(metric_value(q, name)))
            for name in metric_names:
                optimum = minimize_scalar(
                    lambda time, name=name: -float(metric_value(quantities_at(time, segment), name)),
                    bounds=(left, right), method="bounded", options={"xatol": 1e-13},
                )
                maxima[name] = max(maxima[name], -float(optimum.fun))
    return maxima


def test_parabolic_peak_estimator_recovers_a_subgrid_quadratic_vertex():
    x_left, x_center, x_right = 0.25, 0.5, 0.75
    vertex = 0.57
    function = lambda x: 2.0 - (x - vertex) ** 2

    estimated = solver._parabolic_peak(
        np, np.asarray([function(x_left)]), np.asarray([function(x_center)]), np.asarray([function(x_right)])
    )

    np.testing.assert_allclose(estimated, [2.0], rtol=0.0, atol=1e-15)


def test_continuous_peak_diagnostics_are_opt_in_and_do_not_change_canonical_metrics(monkeypatch):
    case = next(case for case in gc.load_corpus()["cases"] if case["id"] == "ratetable-006")
    _, motor, propellant, environment, options = gc.build_objects(case)
    batch = ProblemBatch.from_objects(motor, propellant, environment, options)
    original = solver._continuous_peak_candidates

    def unexpected_diagnostic(*args, **kwargs):
        raise AssertionError("the default solve must not evaluate dense peak diagnostics")

    monkeypatch.setattr(solver, "_continuous_peak_candidates", unexpected_diagnostic)
    ordinary = simulate_burn(batch, backend="cpu-vectorized", strict=True).to_results()[0]
    assert "continuous_peaks" not in ordinary["provenance"]["execution"]

    monkeypatch.setattr(solver, "_continuous_peak_candidates", original)
    diagnostic = simulate_burn(
        batch, backend="cpu-vectorized", strict=True, continuous_peak_diagnostics=True,
    ).to_results()[0]
    assert diagnostic["metrics"] == ordinary["metrics"]
    assert set(diagnostic["provenance"]["execution"]["continuous_peaks"]) == {
        "max_generated_mass_flow_kg_s", "peak_chamber_pressure_pa", "peak_thrust_n",
        "max_nozzle_mass_flow_kg_s", "estimator",
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "backend_name,device,case_id",
    [
        ("cpu-vectorized", None, "ratetable-006"),
        ("cpu-vectorized", None, "guard-thinweb-002"),
        ("cpu-vectorized", None, "ends-tubular-005"),
        pytest.param("jax", "cuda:0", "ratetable-006", marks=pytest.mark.gpu),
        pytest.param("jax", "cuda:0", "guard-thinweb-002", marks=pytest.mark.gpu),
        pytest.param("jax", "cuda:0", "ends-tubular-005", marks=pytest.mark.gpu),
    ],
)
def test_batched_continuous_peaks_match_the_refined_scalar_solution(monkeypatch, backend_name, device, case_id):
    case = next(case for case in gc.load_corpus()["cases"] if case["id"] == case_id)
    refined = _refined_dense_peaks(case, monkeypatch)
    grain, motor, propellant, environment, options = gc.build_objects(case)
    batch = ProblemBatch.from_objects(motor, propellant, environment, options)
    result = simulate_burn(
        batch, backend=backend_name, device=device, strict=True, continuous_peak_diagnostics=True,
    ).to_results()[0]
    continuous = result["provenance"]["execution"]["continuous_peaks"]

    for name, expected in refined.items():
        relative_error = abs(continuous[name] - expected) / abs(expected)
        assert relative_error <= 2e-4, (case_id, name, relative_error, continuous[name], expected)
