import numpy as np
import pytest

import golden_corpus as gc
from solidpy import BurnSimulation, backends
from solidpy.backends import _tolerances as tol
from solidpy.batch import ProblemBatch
from solidpy.batch.kernels import propellant as propellant_kernels
from solidpy.ensemble import lane_cost, simulate_burn
from test_batch_problem import make_stack

FAMILIES = ("tubular", "erosive", "ratetable")  # power law, power law with the erosive term, tabulated rate
FACTORS = (0.94, 1.06, 0.5)
MAX_POINTS = 260


@pytest.fixture(scope="module")
def lanes():
    corpus = gc.load_corpus()["cases"]
    reference = gc.load_reference()["records"]
    cases = []
    for family in FAMILIES:
        members = [c for c in corpus if c["family"] == family and reference[c["id"]]["history_points"] <= MAX_POINTS
                   and reference[c["id"]]["status"]["termination_reason"] == "completed"
                   and {"igniter_none", "activation_none", "scalar_thermo"} <= set(c["tags"])]
        cases.append(min(members, key=lambda c: reference[c["id"]]["history_points"]))
    built = [gc.build_objects(c) for c in cases]
    return cases, built


def pack(built, factors):
    return ProblemBatch.from_objects([b[1] for b in built], [b[2] for b in built], [b[3] for b in built],
                                     [b[4] for b in built], burn_rate_factor=list(factors))


def scalar_with_override(built, factor):
    """The scalar run with the factor applied the way ``Robustness._apply_scenario`` applies it."""
    _, motor, propellant, environment, kwargs = built
    import copy

    propellant = copy.deepcopy(propellant)
    original = propellant.evaluate_burn_rate
    propellant.evaluate_burn_rate = lambda p, g=0.0, _o=original, _f=factor: _f * _o(p, g)
    return BurnSimulation(motor.grains[0], motor, propellant, environment, **kwargs).result


def test_the_factor_defaults_to_one_and_is_broadcast_or_given_per_lane():
    motor, propellant = make_stack()

    default = ProblemBatch.from_objects(motor, propellant)
    single = ProblemBatch.from_objects([motor, motor, motor], propellant, burn_rate_factor=1.06)
    per_lane = ProblemBatch.from_objects(motor, propellant, burn_rate_factor=np.array([0.9, 1.0, 1.1]))
    as_list = ProblemBatch.from_objects(motor, propellant, burn_rate_factor=[0.9, 1.0])

    assert default.arrays["burn_rate_factor"].tolist() == [1.0]
    assert single.arrays["burn_rate_factor"].tolist() == [1.06] * 3
    assert per_lane.arrays["burn_rate_factor"].tolist() == [0.9, 1.0, 1.1] and len(per_lane) == 3
    assert as_list.arrays["burn_rate_factor"].tolist() == [0.9, 1.0] and len(as_list) == 2


@pytest.mark.parametrize("value, message", [
    (0.0, r"lane 0: burn_rate_factor must be a finite positive number"),
    ([1.0, -0.5], r"lane 1: burn_rate_factor must be a finite positive number"),
    (float("nan"), r"lane 0: burn_rate_factor"),
    (float("inf"), r"lane 0: burn_rate_factor"),
    ([[1.0, 1.0]], r"one number per lane"),
    ("fast", r"one number per lane"),
])
def test_an_invalid_factor_is_refused_with_the_lane(value, message):
    motor, propellant = make_stack()

    with pytest.raises(ValueError, match=message):
        ProblemBatch.from_objects([motor, motor], propellant, burn_rate_factor=value)


def test_a_factor_list_that_does_not_fit_the_lanes_is_refused():
    motor, propellant = make_stack()

    with pytest.raises(ValueError, match="burn_rate_factor has 2 entries for 3 lanes"):
        ProblemBatch.from_objects([motor, motor, motor], propellant, burn_rate_factor=[1.0, 1.0])
    with pytest.raises(ValueError, match="entries for 3 lanes"):  # a longer list makes the other inputs the odd ones
        ProblemBatch.from_objects([motor, motor], propellant, burn_rate_factor=[1.0, 1.0, 1.0])


def test_the_kernel_applies_the_factor_to_the_whole_rate_including_the_erosive_term(lanes):
    cases, built = lanes
    batch = pack(built, FACTORS)
    P = batch.namespace(np)
    pressure = np.geomspace(2e5, 8e6, 7)
    flux = np.array([0.0, 5e-4, 2.0, 40.0, 150.0, 400.0, 900.0])

    for lane, (case, objects) in enumerate(zip(cases, built)):
        propellant = objects[2]
        view = {name: array[lane : lane + 1] for name, array in P.items()}
        got = np.array([propellant_kernels.burn_rate(np, np.array([p]), np.array([g]), view)[0]
                        for p in pressure for g in flux])
        expected = np.array([FACTORS[lane] * propellant.evaluate_burn_rate(p, g) for p in pressure for g in flux])
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0, err_msg=case["id"])
    assert any(getattr(b[2], "erosive_burning_coefficient", 0.0) > 0 for b in built)  # the erosive branch was exercised


def test_the_reference_backend_reproduces_the_scenario_override_exactly_and_leaves_the_objects_alone(lanes):
    cases, built = lanes
    batch = pack(built, FACTORS)

    got = backends.get_backend("cpu-reference").solve_burn(batch).to_results()

    for case, objects, factor, result in zip(cases, built, FACTORS, got):
        expected = scalar_with_override(objects, factor)
        assert result["metrics"] == expected["metrics"] and result["status"] == expected["status"], case["id"]
        assert result["provenance"]["physics_provider_hash"] == expected["provenance"]["physics_provider_hash"]
        assert "evaluate_burn_rate" not in vars(objects[2])  # the shared propellant was copied, not patched


def test_the_reference_backend_gives_the_same_results_through_a_process_pool(lanes):
    _, built = lanes
    batch = pack(built[:2], FACTORS[:2])
    reference = backends.get_backend("cpu-reference")

    serial = reference.solve_burn(batch).to_results()
    pooled = reference.solve_burn(batch, backends.SolveOptions(workers=2)).to_results()

    assert [r["metrics"] for r in serial] == [r["metrics"] for r in pooled]


def test_numpy_lanes_match_the_scalar_run_with_the_override(lanes):
    cases, built = lanes
    batch = pack(built, FACTORS)

    got = backends.get_backend("cpu-vectorized").solve_burn(batch).to_results()

    for case, objects, factor, result in zip(cases, built, FACTORS, got):
        expected = scalar_with_override(objects, factor)
        assert result["status"]["termination_reason"] == expected["status"]["termination_reason"], case["id"]
        for key in ("total_impulse_ns", "generated_mass_integral_kg", "nozzle_mass_integral_kg"):
            assert result["metrics"][key] == pytest.approx(expected["metrics"][key], rel=tol.INTEGRAL_RTOL), (case["id"], key)
        for key in ("peak_chamber_pressure_pa", "peak_thrust_n"):
            assert result["metrics"][key] == pytest.approx(expected["metrics"][key], rel=tol.GRID_SAMPLED_RTOL), (case["id"], key)
        for got_t, want_t in zip(result["metrics"]["grain_burnout_times_s"], expected["metrics"]["grain_burnout_times_s"]):
            assert got_t == pytest.approx(want_t, rel=tol.TIME_RTOL), case["id"]
        assert result["provenance"]["physics_provider_hash"] == expected["provenance"]["physics_provider_hash"]
        assert result["provenance"]["execution"]["scenario_inputs"] == {"burn_rate_factor": factor}


def test_a_faster_burn_shortens_the_run_and_a_slower_one_lengthens_it(lanes):
    _, built = lanes
    batch = pack(built[:1] * 3, (0.8, 1.0, 1.25))

    results = backends.get_backend("cpu-vectorized").solve_burn(batch).to_results()
    burnout = [r["metrics"]["grain_burnout_times_s"][0] for r in results]

    assert burnout[0] > burnout[1] > burnout[2]


def test_the_cost_estimate_follows_the_factor(lanes):
    _, built = lanes
    slow, fast = (lane_cost(pack(built[:1], (f,)))[0] for f in (0.5, 2.0))

    assert slow > fast


def test_simulate_burn_runs_factor_lanes_on_the_requested_backend_and_the_fallback(lanes):
    _, built = lanes
    batch = pack(built[:2], (1.06, 0.94))

    on_reference = simulate_burn(batch, backend="cpu-reference").to_results()
    on_numpy = simulate_burn(batch, backend="cpu-vectorized", strict=True).to_results()

    for a, b in zip(on_reference, on_numpy):
        assert a["status"]["termination_reason"] == b["status"]["termination_reason"]
        assert b["metrics"]["total_impulse_ns"] == pytest.approx(a["metrics"]["total_impulse_ns"], rel=tol.INTEGRAL_RTOL)
